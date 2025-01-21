import random
import itertools
from collections import defaultdict
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
        self.current_player = 0 # Assuming the AI is the only player in this implementation

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
        # Implement the logic for fantasy mode bonus calculation
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
    def __init__(self):
        self.nodes = {}
        self.iterations = 100 # Настроечный параметр

    def cfr(self, game_state, p0, p1, timeout_event, result):
        if game_state.is_terminal():
            return game_state.get_payoff()

        player = game_state.get_current_player()
        info_set = game_state.get_information_set()

        if info_set not in self.nodes:
            self.nodes[info_set] = CFRNode(game_state.get_actions())
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

    def train(self, iterations, timeout_event, result):
        for _ in range(iterations):
            # Создание начального состояния игры
            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)

            # Создание пустого состояния игры
            game_state = GameState()

            # Установка карт для игрока (в данном случае, для ИИ)
            game_state.selected_cards = Hand(all_cards[:5])  # Пример: раздача 5 карт для начала

            # Запуск алгоритма CFR
            self.cfr(game_state, 1, 1, timeout_event, result)

    def get_move_5(self, game_state, timeout_event, result):
        # Логика для 5 карт
        best_move = None
        best_value = float('-inf')

        for action in game_state.get_actions():
            if timeout_event.is_set():
                print("Timeout during get_move_5")
                break
            value = self.evaluate_move(game_state, action)
            if value > best_value:
                best_value = value
                best_move = action

        result['move'] = best_move

    def get_move_3(self, game_state, timeout_event, result):
        # Логика для 3 карт
        best_move = None
        best_value = float('-inf')

        for action in game_state.get_actions():
            if timeout_event.is_set():
                print("Timeout during get_move_3")
                break
            value = self.evaluate_move(game_state, action)
            if value > best_value:
                best_value = value
                best_move = action

        result['move'] = best_move

    def get_move_fantasy(self, game_state, timeout_event, result):
        # Логика для режима "Фантазии"
        best_move = None
        best_value = float('-inf')

        for action in game_state.get_actions():
            if timeout_event.is_set():
                print("Timeout during get_move_fantasy")
                break
            value = self.evaluate_move(game_state, action)
            if value > best_value:
                best_value = value
                best_move = action

        result['move'] = best_move

    def evaluate_move(self, game_state, action):
        """Оценивает ход, применяя его к текущему состоянию игры и возвращая ожидаемое значение."""
        next_state = game_state.apply_action(action)
        info_set = next_state.get_information_set()

        if info_set in self.nodes:
            node = self.nodes[info_set]
            strategy = node.get_average_strategy()
            expected_value = 0
            for a, prob in strategy.items():
                # Здесь мы предполагаем, что у нас есть способ получить ценность действия
                # Это может потребовать дальнейшей симуляции или оценки состояния
                action_value = self.get_action_value(next_state, a)
                expected_value += prob * action_value
            return expected_value
        else:
            # Если узел не найден, возвращаем базовую оценку
            return self.baseline_evaluation(next_state)

    def get_action_value(self, state, action):
        """Возвращает ценность действия в данном состоянии.

        Это может быть эвристическая оценка или результат симуляции Монте-Карло.
        """
        # Простая эвристика: оцениваем по количеству очков, которое дает действие
        next_state = state.apply_action(action)
        if next_state.is_terminal():
            return next_state.get_payoff()
        else:
            # Здесь может быть вызов симуляции Монте-Карло или другая эвристика
            return self.baseline_evaluation(next_state)

    def baseline_evaluation(self, state):
        """Базовая эвристическая оценка состояния игры."""
        if state.is_terminal():
            return state.get_payoff()
        else:
            # Простая эвристика: сумма очков за каждую линию
            score = 0
            score += state.get_line_score('top', state.board.top)
            score += state.get_line_score('middle', state.board.middle)
            score += state.get_line_score('bottom', state.board.bottom)
            return score

    def save_progress(self):
        data = {
            'nodes': self.nodes,
            'iterations': self.iterations
        }
        utils.save_data(data, 'cfr_data.pkl')
        github_utils.save_progress_to_github('cfr_data.pkl')

    def load_progress(self):
        github_utils.load_progress_from_github('cfr_data.pkl')
        data = utils.load_data('cfr_data.pkl')
        if data:
            self.nodes = data['nodes']
            self.iterations = data['iterations']

# Создание экземпляра агента
cfr_agent = CFRAgent()
