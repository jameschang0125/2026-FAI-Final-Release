import random
import copy
from collections import defaultdict
import time
import signal

class TimeoutException(BaseException):
    pass

def alarm_handler(signum, frame):
    raise TimeoutException("Timeout exceeded!")

def is_oom(e):
    """True if `e` looks like an out-of-memory condition we must propagate.
    Covers MemoryError directly and POSIX ENOMEM (errno 12) wrapped in OSError.
    """
    return isinstance(e, MemoryError) or (isinstance(e, OSError) and getattr(e, "errno", None) == 12)

class Engine:
    def __init__(self, cfg, players):
        self.n_cards = cfg.get("n_cards", 104)
        self.n_players = cfg.get("n_players", 4)
        self.n_rounds = cfg.get("n_rounds", 10)
        self.board_size_x = cfg.get("board_size_x", 5) # capacity before taking
        self.board_size_y = cfg.get("board_size_y", 4)
        self.verbose = cfg.get("verbose", False)
        self.seed = cfg.get("seed", None)
        self.timeout = cfg.get("timeout", None)
        self.timeout_buffer = cfg.get("timeout_buffer", 0.5)
        self.rng = random.Random(self.seed)
        
        # Mapping from card number to score
        self.score_mapping = self._default_score_mapping()
        
        assert len(players) == self.n_players
        self.players = players
        
        # Optional override for hands and initial board (for duplicate tournaments)
        self.fixed_hands = cfg.get("fixed_hands", None)
        self.fixed_board = cfg.get("fixed_board", None)
        
        # game state
        self.reset()

    def _default_score_mapping(self):
        # Use defaultdict to handle default score of 1
        mapping = defaultdict(lambda: 1)
        for i in range(1, self.n_cards + 1):
            if i % 55 == 0:
                mapping[i] = 7
            elif i % 11 == 0:
                mapping[i] = 5
            elif i % 10 == 0:
                mapping[i] = 3
            elif i % 5 == 0:
                mapping[i] = 2
        return mapping

    def reset(self):
        self.round = 0
        self.scores = [0] * self.n_players
        self.disqualified_players = set()
        self.timeout_counts = defaultdict(int)
        self.exception_counts = defaultdict(int)
        
        # Card deck 1..n_cards. Cards already committed via fixed_hands /
        # fixed_board are excluded so we can never duplicate them.
        deck = list(range(1, self.n_cards + 1))
        dealt = set()
        if self.fixed_hands:
            for hand in self.fixed_hands:
                dealt.update(hand)
        if self.fixed_board:
            for row in self.fixed_board:
                dealt.update(row)
        if dealt:
            deck = [c for c in deck if c not in dealt]
        self.rng.shuffle(deck)

        # Check if we have enough cards
        cards_required = 0
        if not self.fixed_hands:
            cards_required += self.n_players * self.n_rounds
        if not self.fixed_board:
            cards_required += self.board_size_y
        if len(deck) < cards_required:
            raise ValueError(f"Not enough cards! Need {cards_required}, have {len(deck)}")

        if self.fixed_board:
            if len(self.fixed_board) != self.board_size_y:
                raise ValueError(f"fixed_board has {len(self.fixed_board)} rows, expected {self.board_size_y}")
            self.board = [list(row) for row in self.fixed_board]
        else:
            self.board = []
            for _ in range(self.board_size_y):
                card = deck.pop()
                self.board.append([card])

        # History: dynamic lists
        self.history_matrix = []
        self.board_history = []
        self.flags_matrix = []
        self.score_history = []

        # Deal hands: each player gets n_rounds (usually 10) cards
        self.hands = []

        if self.fixed_hands:
             # Use provided hands if available
             # Deepcopy to prevent modification of source
             self.hands = copy.deepcopy(self.fixed_hands)
             if len(self.hands) != self.n_players:
                 raise ValueError(f"fixed_hands has {len(self.hands)} hands, expected {self.n_players}")
        else:
            for player_id in range(self.n_players):
                hand = []
                for _ in range(self.n_rounds):
                    hand.append(deck.pop())
                self.hands.append(sorted(hand))
                
        if self.verbose:
            for i, hand in enumerate(self.hands):
                print(f'Player {i} initalized with cards: {hand}')
        
    def calculate_row_score(self, row):
        return sum(self.score_mapping[c] for c in row)
    
    def process_card_placement(self, card, player_idx):
        """
        Places a card on the board and returns the score incurred.
        """
        # Find row
        best_row_idx = -1
        max_val_under_card = -1
        
        for r_idx, row in enumerate(self.board):
            last_card = row[-1]
            if last_card < card:
                if last_card > max_val_under_card:
                    max_val_under_card = last_card
                    best_row_idx = r_idx
        
        score_incurred = 0
        
        # Case 1: Fits in a row
        if best_row_idx != -1:
            # Check for 6th card (capacity check)
            if len(self.board[best_row_idx]) >= self.board_size_x:
                # Take row
                score_incurred = self.calculate_row_score(self.board[best_row_idx])
                # New row starts with this card
                self.board[best_row_idx] = [card]
                if self.verbose:
                    print(f"Player {player_idx} takes row {best_row_idx} ({score_incurred} pts) with card {card}")
            else:
                self.board[best_row_idx].append(card)
                if self.verbose:
                    print(f"Player {player_idx} plays card {card} on row {best_row_idx}")
        
        # Case 2: Lower than all rows (Low Card Rule)
        else:
            # Choose row to take based on rules:
            # 1. Fewest score points
            # 2. Shortest len
            # 3. Lowest index
            
            chosen_r_idx = min(range(len(self.board)), key=lambda i: (self.calculate_row_score(self.board[i]), len(self.board[i]), i))
            score_incurred = self.calculate_row_score(self.board[chosen_r_idx])
            
            self.board[chosen_r_idx] = [card]
            
            if self.verbose:
                print(f"Player {player_idx} plays low card {card}, takes row {chosen_r_idx} ({score_incurred} pts)")
        
        self.scores[player_idx] += score_incurred
        return score_incurred

    def play_round(self):
        # Snapshot current board state
        self.board_history.append([row.copy() for row in self.board])
        
        if self.verbose:
            print(f"\n=== Round {self.round + 1} ===")
            print(f"Scores: {self.scores}")
            print("Board:")
            for row in self.board:
                print(f"  {row}")
        
        current_played_cards = [] # (card, player_index)
        
        # Vectors for this round
        round_actions = [0] * self.n_players
        round_flags = [False] * self.n_players  # True = interrupted
        
        # 1. Collect actions
        history_state = {
            "board": self.board,
            "scores": self.scores,
            "round": self.round,
            "history_matrix": self.history_matrix,
            "board_history": self.board_history,
            "score_history": self.score_history,
        }
        
        for p_idx, player in enumerate(self.players):
            if self.verbose:
                print(f"Player {p_idx} taking action...")
            hand = self.hands[p_idx]
            
            is_forced = False
            played_card = None
            
            if p_idx in self.disqualified_players:
                # replace with smallest card
                played_card = hand[0]
                is_forced = True
                if self.verbose:
                    print(f"Player {p_idx} is disqualified. Playing smallest card: {played_card}")
            else:
                start_time = time.time()
                if self.timeout:
                    signal.signal(signal.SIGALRM, alarm_handler)
                    signal.setitimer(signal.ITIMER_REAL, self.timeout)
                    
                try:
                    # Reset global random state before each action to prevent
                    # players from manipulating each other's random streams.
                    random.seed(None)
                    # Pass copy of hand to prevent modification
                    # Pass DEEP copy of history to prevent modification
                    played_card = player.action(hand.copy(), copy.deepcopy(history_state))
                            
                except TimeoutException:
                    if self.verbose:
                        print(f"Player {p_idx} timed out! Defaulting to smallest card.")
                    self.timeout_counts[p_idx] += 1
                    played_card = hand[0]
                    is_forced = True
                except Exception as e:
                    if is_oom(e):
                        raise
                    if self.verbose:
                        print(f"Player {p_idx} crashed: {e}")
                    self.exception_counts[p_idx] += 1
                    played_card = hand[0]
                    is_forced = True
                finally:
                    try:
                        if self.timeout:
                            signal.signal(signal.SIGALRM, signal.SIG_IGN)
                            signal.setitimer(signal.ITIMER_REAL, 0)
                    except TimeoutException:
                        pass
                
                elapsed = time.time() - start_time
                if self.timeout and elapsed > self.timeout + self.timeout_buffer:
                    # They caught the TimeoutException and swallowed it!
                    self.disqualified_players.add(p_idx)
                    played_card = hand[0]
                    is_forced = True
                    if self.verbose:
                        print(f"Player {p_idx} swallowed timeout exception! Disqualified and played smallest card: {played_card}")

            # Validation
            if not is_forced and (not isinstance(played_card, int) or played_card not in hand):
                # Fallback: pick smallest
                played_card = hand[0]
                is_forced = True
                if self.verbose:
                    print(f"Player {p_idx} tried to play invalid card! Defaulting to {played_card}")
            
            hand.remove(played_card)
            current_played_cards.append((played_card, p_idx))
            
            # Record action
            round_actions[p_idx] = played_card
            round_flags[p_idx] = is_forced
            
        # Update history
        self.history_matrix.append(round_actions)
        self.flags_matrix.append(round_flags)

        # 2. Sort by card value
        current_played_cards.sort(key=lambda x: x[0])
        
        # 3. Process placements
        for card, p_idx in current_played_cards:
            self.process_card_placement(card, p_idx)
        self.score_history.append(list(self.scores))

    def play_game(self):
        for _ in range(self.n_rounds):
            self.play_round()
            self.round += 1
            
        full_history = {
            "board_history": self.board_history,
            "flags_matrix": self.flags_matrix,
            "final_scores": self.scores,
            "disqualified_players": list(self.disqualified_players),
            "timeout_counts": dict(self.timeout_counts),
            "exception_counts": dict(self.exception_counts)
        }
        return self.scores, full_history

    # these are helper functions that is not used in the main playing loop
    def clone(self, players=None):
        """Create a new Engine with copied state. Optionally replace players."""
        cfg = {
            "n_cards": self.n_cards, "n_players": self.n_players, "n_rounds": self.n_rounds,
            "board_size_x": self.board_size_x, "board_size_y": self.board_size_y,
            "verbose": False, "seed": None,
        }
        new_engine = Engine(cfg, players or list(self.players))
        new_engine.round = self.round
        new_engine.scores = [s for s in self.scores]
        new_engine.board = [row.copy() for row in self.board]
        new_engine.hands = [h.copy() for h in self.hands]
        new_engine.history_matrix = [r.copy() for r in self.history_matrix]
        new_engine.board_history = [b.copy() for b in self.board_history]
        new_engine.flags_matrix = [f.copy() for f in self.flags_matrix]
        new_engine.score_history = [s.copy() for s in self.score_history]
        new_engine.exception_counts = defaultdict(int, self.exception_counts)
        return new_engine

    def play_remaining(self):
        """Play from current round until game end. Returns scores and history."""
        while self.round < self.n_rounds:
            self.play_round()
            self.round += 1
        return self.scores, {
            "board_history": self.board_history,
            "history_matrix": self.history_matrix,
            "flags_matrix": self.flags_matrix,
            "final_scores": self.scores
        }