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

# Global AI agent instance
cfr_agent = None

# Function to initialize the AI agent with settings
def initialize_ai_agent(ai_settings):
    global cfr_agent
    iterations = int(ai_settings.get('iterations', 1000))
    stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    cfr_agent = ai_engine.CFRAgent(iterations=iterations, stop_threshold=stop_threshold)

    if os.environ.get("AI_PROGRESS_TOKEN"):
        try:
            cfr_agent.load_progress()
            print("AI progress loaded successfully.")
        except Exception as e:
            print(f"Error loading AI progress: {e}")
    else:
        print("AI_PROGRESS_TOKEN not set. Progress loading disabled.")

# Initialize AI agent with default settings on app start
initialize_ai_agent({})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/training')
def training():
    # Initialize game state if it doesn't exist
    if 'game_state' not in session:
        session['game_state'] = {
            'selected_cards': [],
            'board': {'top': [], 'middle': [], 'bottom': []},
            'discarded_cards': [],
            'ai_settings': {
                'fantasyType': 'normal',
                'fantasyMode': False,
                'aiTime': '5',
                'iterations': '1000',
                'stopThreshold': '0.001',
                'aiType': 'mccfr'
            }
        }
    # Initialize AI agent if it's not initialized or settings have changed
    if cfr_agent is None or session['game_state']['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()
    return render_template('training.html', game_state=session['game_state'])


@app.route('/update_state', methods=['POST'])
def update_state():
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400

    game_state = request.get_json()

    if not isinstance(game_state, dict):
        return jsonify({'error': 'Invalid game state format'}), 400

    # Update game state in session
    session['game_state'] = game_state
    session.modified = True # Ensure session is saved

    # Reinitialize AI agent if settings have changed
    if game_state['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(game_state['ai_settings'])
        session['previous_ai_settings'] = game_state['ai_settings'].copy()

    return jsonify({'status': 'success'})


@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent
    game_state_data = request.get_json()
    print("Received game_state_data:", game_state_data)

    num_cards = len(game_state_data['selected_cards'])
    ai_settings = game_state_data['ai_settings']

    selected_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data['selected_cards']]
    board = ai_engine.Board()
    for line in ['top', 'middle', 'bottom']:
        for card_data in game_state_data['board'].get(line, []):
            board.place_card(line, ai_engine.Card(card_data['rank'], card_data['suit']))
    try:
        discarded_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data['discarded_cards']]
    except KeyError as e:
        print(f"KeyError: {e} not found in game_state_data['discarded_cards']")
        print("game_state_data['discarded_cards']:", game_state_data['discarded_cards'])
        return jsonify({'error': f"KeyError: {e} not found in discarded_cards"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': str(e)}), 500

    game_state = ai_engine.GameState(selected_cards=selected_cards, board=board, discarded_cards=discarded_cards, ai_settings=ai_settings)

    timeout_event = Event()
    result = {}

    def worker():
        try:
            cfr_agent.get_move(game_state, num_cards, timeout_event, result)
        except Exception as e:
            result['move'] = {'error': str(e)}

    thread = Thread(target=worker)
    thread.start()
    thread.join(timeout=float(ai_settings['aiTime']))
    timeout_event.set()

    # Проверка, успел ли поток завершиться
    if thread.is_alive():
        print("AI move timed out!")
        return jsonify({'error': 'AI move timed out'}), 500

    # Проверка на наличие результата и ошибок
    if 'move' not in result or (result and 'error' in result.get('move', {})):
        error_message = result.get('move', {}).get('error', 'Unknown error occurred during AI move')
        return jsonify({'error': error_message}), 500

    move = result['move']

    # Update game state in session
    session['game_state']['board'] = {
        'top': [{'rank': card.rank, 'suit': card.suit} for card in move['top']],
        'middle': [{'rank': card.rank, 'suit': card.suit} for card in move['middle']],
        'bottom': [{'rank': card.rank, 'suit': card.suit} for card in move['bottom']]
    }
    if move.get('discarded'):
        session['game_state']['discarded_cards'].extend([{'rank': card.rank, 'suit': card.suit} for card in move['discarded']])
    session.modified = True

    # Save AI progress periodically (e.g., every 100 iterations)
    if cfr_agent.iterations % 100 == 0:
        try:
            cfr_agent.save_progress()
            print("AI progress saved successfully.")
        except Exception as e:
            print(f"Error saving AI progress: {e}")

    return jsonify(move)


if __name__ == '__main__':
    app.run(debug=True)
