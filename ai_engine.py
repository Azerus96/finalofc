import random
import itertools
from collections import defaultdict
from github import GithubException
import github_utils
import utils
from threading import Thread, Event
import time

class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']

    def __init__(self, rank, suit):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_dict(self):
        return {'rank': self.rank, 'suit': self.suit}

    @staticmethod
    def get_all_cards():
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.remove(card)

    def __repr__(self):
        return ', '.join(map(str, self.cards))

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

class Board:
    def __init__(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def place_card(self, line, card):
        if line == 'top':
            if len(self.top) >= 3:
                raise ValueError("Top line is full")
            self.top.append(card)
        elif line == 'middle':
            if len(self.middle) >= 5:
                raise ValueError("Middle line is full")
            self.middle.append(card)
        elif line == 'bottom':
            if len(self.bottom) >= 5:
                raise ValueError("Bottom line is full")
            self.bottom.append(card)
        else:
            raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")

    def is_full(self):
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self):
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line):
        if line == 'top':
            return self.top
        elif line == 'middle':
            return self.middle
        elif line == 'bottom':
            return self.bottom
        else:
            raise ValueError("Invalid line specified")

class GameState:
    def __init__(self, selected_cards=None, board=None, discarded_cards=None, ai_settings=None):
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0  # Assuming the AI is the only player in this implementation

    def get_current_player(self):
        return self.current_player

    def get_actions(self):
        """Returns the valid actions for the current state."""
        num_cards = len(self.selected_cards)
        actions = []

        if num_cards == 5:
            # Generate all possible permutations for placing 5 cards
            for p in itertools.permutations(self.selected_cards.cards):
                actions.append({
                    'top': [p[0]],
                    'middle': [p[1], p[2]],
                    'bottom': [p[3], p[4]],
                    'discarded': None
                })
        elif num_cards == 3:
            # Generate all possible permutations for placing 2 cards and discarding 1
            for p in itertools.permutations(self.selected_cards.cards):
                actions.append({
                    'top': [p[0]],
                    'middle': [p[1]],
                    'bottom': [],
                    'discarded': p[2]
                })
                actions.append({
                    'top': [p[0]],
                    'middle': [],
                    'bottom': [p[1]],
                    'discarded': p[2]
                })
                actions.append({
                    'top': [],
                    'middle': [p[0]],
                    'bottom': [p[1]],
                    'discarded': p[2]
                })
        elif num_cards >= 13:
            # Generate permutations for fantasy mode
            for p in itertools.permutations(self.selected_cards.cards):
                actions.append({
                    'top': list(p[:3]),
                    'middle': list(p[3:8]),
                    'bottom': list(p[8:13]),
                    'discarded': list(p[13:])
                })

        return actions

    def is_terminal(self):
        """Checks if the current state is a terminal state."""
        return self.board.is_full()

    def apply_action(self, action):
        """Applies an action to the current state and returns the new state."""
        new_board = Board()
        new_discarded_cards = self.discarded_cards[:]

        # Place cards on the new board
        if action['top']:
            for card in action['top']:
                new_board.place_card('top', card)
        if action['middle']:
            for card in action['middle']:
                new_board.place_card('middle', card)
        if action['bottom']:
            for card in action['bottom']:
                new_board.place_card('bottom', card)

        # Handle discarded card
        if action['discarded'] is not None:
            if isinstance(action['discarded'], list):
                new_discarded_cards.extend(action['discarded'])
            else:
                new_discarded_cards.append(action['discarded'])

        # Create a new GameState with the updated board and discarded cards
        new_game_state = GameState(
            selected_cards=Hand(),  # Assuming all selected cards are used
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings
        )

        return new_game_state

    def get_information_set(self):
        """Returns a string representation of the current information set."""
        top_str = ','.join(map(str, sorted(self.board.top, key=lambda c: (Card.RANKS.index(c.rank), Card.SUITS.index(c.suit)))))
        middle_str = ','.join(map(str, sorted(self.board.middle, key=lambda c: (Card.RANKS.index(c.rank), Card.SUITS.index(c.suit)))))
        bottom_str = ','.join(map(str, sorted(self.board.bottom, key=lambda c: (Card.RANKS.index(c.rank), Card.SUITS.index(c.suit)))))
        discarded_str = ','.join(map(str, sorted(self.discarded_cards, key=lambda c: (Card.RANKS.index(c.rank), Card.SUITS.index(c.suit)))))

        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}"

    def get_payoff(self):
        """Calculates the payoff for the current state."""
        if not self.is_terminal():
            raise ValueError("Game is not in a terminal state")

        # Check for dead hand
        if self.is_dead_hand():
            return -self.calculate_score()  # Negative score for a dead hand

        # Calculate score if not a dead hand
        return self.calculate_score()

    def is_dead_hand(self):
        """Checks if the hand is a dead hand (invalid combination order)."""
        if not self.board.is_full():
            return False

        top_rank = self.evaluate_hand(self.board.top)
        middle_rank = self.evaluate_hand(self.board.middle)
        bottom_rank = self.evaluate_hand(self.board.bottom)

        # Check if the hand is dead
        return top_rank > middle_rank or middle_rank > bottom_rank

    def calculate_score(self):
        """Calculates the score for the current state based on the rules."""
        score = 0

        # Add scores for each line based on hand rankings
        score += self.get_line_score('top', self.board.top)
        score += self.get_line_score('middle', self.board.middle)
        score += self.get_line_score('bottom', self.board.bottom)

        # Add bonus for fantasy mode if applicable
        if self.ai_settings.get('fantasyMode', False):
            score += self.get_fantasy_bonus()

        return score

    def get_line_score(self, line, cards):
        """Calculates the score for a specific line based on hand rankings."""
        if not cards:
            return 0

        rank = self.evaluate_hand(cards)
        score = 0

        if line == 'top' and len(cards) == 3:
            if rank == 1:  # Royal Flush
                score = 25
            elif rank == 2:  # Straight Flush
                score = 15
            elif rank == 3:  # Four of a Kind
                score = 10
            elif rank == 4:  # Full House
                score = 6
            elif rank == 5:  # Flush
                score = 4
            elif rank == 6:  # Straight
                score = 2
            elif rank == 7:  # Three of a Kind
                score = 2
            elif rank == 8:
                score = self.get_pair_bonus(cards)
            elif rank == 9:
                score = self.get_high_card_bonus(cards)

        elif line == 'middle' and len(cards) == 5:
            if rank == 1:  # Royal Flush
                score = 50
            elif rank == 2:  # Straight Flush
                score = 30
            elif rank == 3:  # Four of a Kind
                score = 20
            elif rank == 4:  # Full House
                score = 12
            elif rank == 5:  # Flush
                score = 8
            elif rank == 6:  # Straight
                score = 4
            elif rank == 7:  # Three of a Kind
                score = 2

        elif line == 'bottom' and len(cards) == 5:
            if rank == 1:  # Royal Flush
                score = 25
            elif rank == 2:  # Straight Flush
                score = 15
            elif rank == 3:  # Four of a Kind
                score = 10
            elif rank == 4:  # Full House
                score = 6
            elif rank == 5:  # Flush
                score = 4
            elif rank == 6:  # Straight
                score = 2

        return score

    def get_pair_bonus(self, cards):
        """Calculates the bonus for a pair in the top line."""
        if len(cards) != 3:
            return 0
        ranks = [card.rank for card in cards]
        for rank in Card.RANKS[::-1]:
            if ranks.count(rank) == 2:
                if rank == '6':
                    return 1
                elif rank == '7':
                    return 2
                elif rank == '8':
                    return 3
                elif rank == '9':
                    return 4
                elif rank == '10':
                    return 5
                elif rank == 'J':
                    return 6
                elif rank == 'Q':
                    return 7
                elif rank == 'K':
                    return 8
                elif rank == 'A':
                    return 9
        return 0

    def get_high_card_bonus(self, cards):
        """Calculates the bonus for a high card in the top line."""
        if len(cards) != 3:
            return 0
        ranks = [card.rank for card in cards]
        if len(set(ranks)) == 3:  # Three different ranks
            high_card = max(ranks, key=Card.RANKS.index)
            if high_card == 'A':
                return 1
        return 0

    def get_fantasy_bonus(self):
        """Calculates the bonus for fantasy mode."""
        # Placeholder for fantasy mode bonus calculation
        # TODO: Implement the actual logic for fantasy mode bonus calculation
        return 0

    def evaluate_hand(self, cards):
        """Evaluates the hand and returns a rank (lower is better)."""
        if len(cards) == 5:
            # Check for Royal Flush
            if self.is_royal_flush(cards):
                return 1
            # Check for Straight Flush
            if self.is_straight_flush(cards):
                return 2
            # Check for Four of a Kind
            if self.is_four_of_a_kind(cards):
                return 3
            # Check for Full House
            if self.is_full_house(cards):
                return 4
            # Check for Flush
            if self.is_flush(cards):
                return 5
            # Check for Straight
            if self.is_straight(cards):
                return 6
            # Check for Three of a Kind
            if self.is_three_of_a_kind(cards):
                return 7
            # Check for Two Pair
            if self.is_two_pair(cards):
                return 8
            # Check for One Pair
            if self.is_one_pair(cards):
                return 9
            # High Card
            return 10

        elif len(cards) == 3:
            # Check for Three of a Kind
            if self.is_three_of_a_kind(cards):
                return 7
            # Check for One Pair
            if self.is_one_pair(cards):
                return 8
            # High Card
            return 9

        else:
            return 0

    # Helper functions for hand evaluation
    def is_royal_flush(self, cards):
        if not self.is_flush(cards):
            return False
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        return ranks == [8, 9, 10, 11, 12]  # 10, J, Q, K, A

    def is_straight_flush(self, cards):
        return self.is_straight(cards) and self.is_flush(cards)

    def is_four_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 4 for r in ranks)

    def is_full_house(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks) and any(ranks.count(r) == 2 for r in ranks)

    def is_flush(self, cards):
        suits = [card.suit for card in cards]
        return len(set(suits)) == 1

    def is_straight(self, cards):
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        if ranks == [0, 1, 2, 3, 12]:  # Special case for A, 2, 3, 4, 5
            return True
        return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))

    def is_three_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks)

    def is_two_pair(self, cards):
        ranks = [card.rank for card in cards]
        pairs = [r for r in set(ranks) if ranks.count(r) == 2]
        return len(pairs) == 2

    def is_one_pair(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 2 for r in ranks)


class CFRNode:
    def __init__(self, actions):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        strategy = defaultdict(float)
        for a in self.actions:
            strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += strategy[a]

        for a in self.actions:
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / len(self.actions)
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self):
        avg_strategy = defaultdict(float)
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            for a in self.actions:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
        else:
            for a in self.actions:
                avg_strategy[a] = 1.0 / len(self.actions)
        return avg_strategy


class CFRAgent:
    def __init__(self, iterations=1000, stop_threshold=0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold

    def cfr(self, game_state, p0, p1, timeout_event, result):
        if timeout_event.is_set():
            return 0  # Return 0 if timeout occurred

        if game_state.is_terminal():
            return game_state.get_payoff()

        player = game_state.get_current_player()
        info_set = game_state.get_information_set()

        if info_set not in self.nodes:
            actions = game_state.get_actions()
            if not actions:
                return 0 # Return 0 if no actions are available
            self.nodes[info_set] = CFRNode(actions)
        node = self.nodes[info_set]

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = defaultdict(float)
        node_util = 0

        for a in node.actions:
            next_state = game_state.apply_action(a)
            if player == 0:
                util[a] = -self.cfr(next_state, p0 * strategy[a], p1, timeout_event, result)
            else:
                util[a] = -self.cfr(next_state, p0, p1 * strategy[a], timeout_event, result)
            node_util += strategy[a] * util[a]

        if player == 0:
            for a in node.actions:
                node.regret_sum[a] += p1 * (util[a] - node_util)
        else:
            for a in node.actions:
                node.regret_sum[a] += p0 * (util[a] - node_util)

        return node_util

    def train(self, timeout_event, result):
        for i in range(self.iterations):
            if timeout_event.is_set():
                break
            # Создание начального состояния игры
            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)

            # Создание пустого состояния игры
            game_state = GameState()

            # Установка карт для игрока (в данном случае, для ИИ)
            game_state.selected_cards = Hand(all_cards[:5])  # Пример: раздача 5 карт для начала

            # Запуск алгоритма CFR
            self.cfr(game_state, 1, 1, timeout_event, result)

            # Check for convergence every 100 iterations
            if i % 100 == 0:
                if self.check_convergence():
                    print("CFR agent converged after", i, "iterations.")
                    break


    def check_convergence(self):
        """Checks if the average strategy has converged."""
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            for action, prob in avg_strategy.items():
                if abs(prob - 1.0 / len(node.actions)) > self.stop_threshold:
                    return False
        return True


    
    def get_move(self, game_state, num_cards, timeout_event, result):
        """Gets the AI's move for a given number of cards."""
        print("Inside get_move")
        actions = game_state.get_actions()
        print("Actions:", actions)  # Добавлено для отладки

        if not actions:
            result['move'] = {'error': 'Нет доступных ходов'}
            return

        # Упрощение: выбираем случайный ход
        best_move = random.choice(actions) if actions else None

        # Всегда устанавливаем result['move'], даже если best_move is None
        result['move'] = best_move

        print("Result['move'] inside get_move:", result['move']) # Добавьте этот print для отладки

    def evaluate_move(self, game_state, action, timeout_event):
        try:
            next_state = game_state.apply_action(action)
            info_set = next_state.get_information_set()

            if info_set in self.nodes:
                node = self.nodes[info_set]
                strategy = node.get_average_strategy()
                expected_value = 0
                for a, prob in strategy.items():
                    try:
                        action_value = self.get_action_value(next_state, a, timeout_event)
                    except Exception as e:
                        print(f"Error in get_action_value within evaluate_move: {e}")
                        raise # Передаем исключение дальше
                    expected_value += prob * action_value
                return expected_value
            else:
                # If the node is not found, perform a shallow search
                return self.shallow_search(next_state, 2, timeout_event) # Search depth of 2
        except Exception as e:
            print(f"Ошибка в evaluate_move: {e}")
            raise # Передаем исключение дальше


    def shallow_search(self, state, depth, timeout_event):
        try:
            if depth == 0 or state.is_terminal() or timeout_event.is_set():
                return self.baseline_evaluation(state)

            best_value = float('-inf')
            for action in state.get_actions():
                value = -self.shallow_search(state.apply_action(action), depth - 1, timeout_event)
                best_value = max(best_value, value)
            return best_value
        except Exception as e:
            print(f"Ошибка в shallow_search: {e}")
            raise # Передаем исключение дальше


    def get_action_value(self, state, action, timeout_event):
        """Returns the value of an action in a given state using Monte Carlo simulation."""
        num_simulations = 10  # Number of Monte Carlo simulations
        total_score = 0

        for _ in range(num_simulations):
            if timeout_event.is_set():
                break
            simulated_state = state.apply_action(action)
            while not simulated_state.is_terminal():
                actions = simulated_state.get_actions()
                if not actions:
                    break
                random_action = random.choice(actions)
                simulated_state = simulated_state.apply_action(random_action)
            total_score += self.baseline_evaluation(simulated_state)

        return total_score / num_simulations if num_simulations > 0 else 0


    def baseline_evaluation(self, state):
        """Baseline heuristic evaluation of the game state."""
        if state.is_dead_hand():
            return -1000 # Large negative penalty for dead hands

        score = 0
        score += state.get_line_score('top', state.board.top)
        score += state.get_line_score('middle', state.board.middle)
        score += state.get_line_score('bottom', state.board.bottom)

        # Add some logic to favor better combinations on higher lines
        score += sum(Card.RANKS.index(card.rank) for card in state.board.top) * 0.5
        score += sum(Card.RANKS.index(card.rank) for card in state.board.middle) * 0.3
        score += sum(Card.RANKS.index(card.rank) for card in state.board.bottom) * 0.2

        return score

    def save_progress(self):
        data = {
            'nodes': self.nodes,
            'iterations': self.iterations,
            'stop_threshold': self.stop_threshold
        }
        utils.save_data(data, 'cfr_data.pkl')
        github_utils.save_progress_to_github('cfr_data.pkl')

    def load_progress(self):
        github_utils.load_progress_from_github('cfr_data.pkl')
        data = utils.load_data('cfr_data.pkl')
        if data:
            self.nodes = data['nodes']
            self.iterations = data['iterations']
            self.stop_threshold = data.get('stop_threshold', 0.001) # Default value if not present


# Creating an instance of the agent
cfr_agent = CFRAgent()
