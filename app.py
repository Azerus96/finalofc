from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
import utils
import github_utils
import time
import json
from threading import Thread, Event

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Загрузка прогресса ИИ при запуске приложения
if os.environ.get("AI_PROGRESS_TOKEN"):
    try:
        ai_engine.cfr_agent.load_progress()
        print("Прогресс ИИ успешно загружен.")
    except Exception as e:
        print(f"Ошибка при загрузке прогресса ИИ: {e}")
else:
    print("Переменная AI_PROGRESS_TOKEN не установлена. Загрузка прогресса невозможна.")

@app.route('/')
def home():
    # Стартовая страница (не используется в данной реализации)
    return render_template('index.html')

@app.route('/training')
def training():
    # Инициализация состояния игры, если оно еще не создано
    if 'game_state' not in session:
        session['game_state'] = {
            'selected_cards': [],
            'board': {
                'top': [],
                'middle': [],
                'bottom': []
            },
            'discarded_cards': [],
            'ai_settings': {
                'fantasyType': 'normal',
                'fantasyMode': False,
                'aiTime': '5',
                'aiType': 'mccfr'
            }
        }
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400

    game_state = request.get_json()

    if not isinstance(game_state, dict):
        return jsonify({'error': 'Invalid game state format'}), 400

    # Обновление состояния игры в сессии
    session['game_state'] = game_state
    return jsonify({'status': 'success'})

@app.route('/ai_move', methods=['POST'])
def ai_move():
    game_state_data = request.get_json()
    num_cards = len(game_state_data['selected_cards'])
    ai_settings = game_state_data['ai_settings']

    # Преобразование данных из JSON в объекты Card
    selected_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data['selected_cards']]
    board = ai_engine.Board()
    if game_state_data['board']['top']:
        for card_data in game_state_data['board']['top']:
            board.place_card('top', ai_engine.Card(card_data['rank'], card_data['suit']))
    if game_state_data['board']['middle']:
        for card_data in game_state_data['board']['middle']:
            board.place_card('middle', ai_engine.Card(card_data['rank'], card_data['suit']))
    if game_state_data['board']['bottom']:
        for card_data in game_state_data['board']['bottom']:
            board.place_card('bottom', ai_engine.Card(card_data['rank'], card_data['suit']))

    discarded_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data['discarded_cards']]

    # Создание объекта GameState
    game_state = ai_engine.GameState(selected_cards=selected_cards, board=board, discarded_cards=discarded_cards, ai_settings=ai_settings)

    # Ограничение времени на ход ИИ
    timeout_event = Event()
    result = {}

    def worker():
      try:
        if num_cards == 5:
            ai_engine.cfr_agent.get_move_5(game_state, timeout_event, result)
        elif num_cards == 3:
            ai_engine.cfr_agent.get_move_3(game_state, timeout_event, result)
        elif num_cards >= 13:
            ai_engine.cfr_agent.get_move_fantasy(game_state, timeout_event, result)
        else:
            result['move'] = {'error': 'Неверное количество карт'}
      except Exception as e:
        result['move'] = {'error': str(e)}

    thread = Thread(target=worker)
    thread.start()
    thread.join(timeout=float(ai_settings['aiTime']))

    if not timeout_event.is_set():
        print("Превышено время ожидания хода ИИ")
        result['move'] = {'error': 'Превышено время ожидания хода ИИ'}
    
    if 'error' in result.get('move', {}):
        return jsonify(result['move'])

    # Обновление состояния игры в сессии
    session['game_state']['board'] = {
        'top': [{'rank': card.rank, 'suit': card.suit} for card in result['move']['top']],
        'middle': [{'rank': card.rank, 'suit': card.suit} for card in result['move']['middle']],
        'bottom': [{'rank': card.rank, 'suit': card.suit} for card in result['move']['bottom']]
    }
    session['game_state']['discarded_cards'].extend([{'rank': card.rank, 'suit': card.suit} for card in result['move'].get('discarded', [])])
    session.modified = True

    # Сохранение прогресса ИИ после каждого хода
    try:
        ai_engine.cfr_agent.save_progress()
        print("Прогресс ИИ успешно сохранен.")
    except Exception as e:
        print(f"Ошибка при сохранении прогресса ИИ: {e}")

    return jsonify(result['move'])

if __name__ == '__main__':
    app.run(debug=True)
