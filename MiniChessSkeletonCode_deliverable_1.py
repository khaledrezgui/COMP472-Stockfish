import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()
        self.no_capture_turns = 0  # Track turns without a capture
        self.total_half_turns = 0 
        self.nodes_explored = 0 # Track total nodes explored in Minimax

        self.board_history = {}
        self.depth_stats = {}  # Track number of states explored per depth
        self.total_branching = 0  # Total number of branches explored (for avg branching factor)
        self.total_nodes = 0  # Total number of nodes (excluding root)
    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
                "board": 
                [['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']],
                "turn": 'white',
                }
        return state

    """
    Prints the board
    
    Args:
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()
    '''
    Next functions are used to prevent AI V AI from repetitve redundant moves
    
    '''
    def hash_board(self, game_state):
        '''Generates a unique hash for the current board state to detect repetition'''
        return tuple(tuple(row) for row in game_state["board"])

    def is_repetition(self):
        """
        Detects if the game has entered a repetition cycle (3 times).
        If repetition occurs, penalizes the AI and forces a different move.
        """
        board_hash = self.hash_board(self.current_game_state)
        self.board_history[board_hash] = self.board_history.get(board_hash, 0) + 1

        if self.board_history[board_hash] >= 3:
            print("Repetition detected! Penalizing AI and forcing different moves.")
            return True

        return False

    def is_terminal(self, game_state):
        """
        Checks if the game is in a terminal state.
        Returns a tuple (True/False, winner or None)
        """
        # Check if any King is missing
        kings = {piece for row in game_state["board"] for piece in row if piece in ("wK", "bK")}
        if "wK" not in kings:
            game_state["winner"] = "black"
            return True, "black"
        if "bK" not in kings:
            game_state["winner"] = "white"
            return True, "white"

        # Check for no valid moves (stalemate or loss)
        if not self.valid_moves(game_state):
            game_state["winner"] = "draw"  # Treat as draw for now (you can customize this logic)
            return True, "draw"

        # Check draw by inactivity (no captures in 10 full turns)
        if (self.no_capture_turns // 2) >= 10:
            game_state["winner"] = "draw"
            return True, "draw"

        # Check repetition
        if self.is_repetition():
            game_state["winner"] = "draw"
            return True, "draw"

        return False, None

    def evaluate_board(self, game_state):
        """
        Evaluates the board using heuristic e0.
        A positive value favors white, a negative value favors black.
        """
        piece_values = {'p': 1, 'B': 3, 'N': 3, 'Q': 9, 'K': 999}
        score = 0

        for row in game_state["board"]:
            for piece in row:
                if piece != '.':
                    value = piece_values[piece[1]]  # Get piece value
                    score += value if piece[0] == 'w' else -value

        return score

    # Minimax Algorithm with Alpha-Beta Pruning
    def minimax(self, game_state, depth, alpha, beta, maximizing_player, current_depth=0):
        self.nodes_explored += 1
        self.depth_stats[current_depth] = self.depth_stats.get(current_depth, 0) + 1

        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate_board(game_state), None

        valid_moves = self.valid_moves(game_state)
        self.total_branching += len(valid_moves)
        self.total_nodes += 1

        best_move = None

        if maximizing_player:
            value = -math.inf
            for move in valid_moves:
                simulated_state = copy.deepcopy(game_state)
                self.make_move(simulated_state, move, simulated=True)
                eval_score, _ = self.minimax(simulated_state, depth - 1, alpha, beta, False, current_depth + 1)

                if eval_score > value:
                    value = eval_score
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move
        else:
            value = math.inf
            for move in valid_moves:
                simulated_state = copy.deepcopy(game_state)
                self.make_move(simulated_state, move, simulated=True)
                eval_score, _ = self.minimax(simulated_state, depth - 1, alpha, beta, True, current_depth + 1)

                if eval_score < value:
                    value = eval_score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    # Minimax without alpha-beta pruning
    def minimax_without_pruning(self, game_state, depth, maximizing_player, current_depth=0):
        self.nodes_explored += 1
        self.depth_stats[current_depth] = self.depth_stats.get(current_depth, 0) + 1

        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate_board(game_state), None

        valid_moves = self.valid_moves(game_state)
        self.total_branching += len(valid_moves)
        self.total_nodes += 1

        best_move = None

        if maximizing_player:
            value = -math.inf
            for move in valid_moves:
                simulated_state = copy.deepcopy(game_state)
                self.make_move(simulated_state, move, simulated=True)
                eval_score, _ = self.minimax_without_pruning(simulated_state, depth - 1, False, current_depth + 1)

                if eval_score > value:
                    value = eval_score
                    best_move = move

            return value, best_move
        else:
            value = math.inf
            for move in valid_moves:
                simulated_state = copy.deepcopy(game_state)
                self.make_move(simulated_state, move, simulated=True)
                eval_score, _ = self.minimax_without_pruning(simulated_state, depth - 1, True, current_depth + 1)

                if eval_score < value:
                    value = eval_score
                    best_move = move

            return value, best_move
    
            return value, best_move
    """
    Check if the move is valid    
    
    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """

    def is_valid_move(self, game_state, move):
        """
        Checks if the move is legal based on:
        - If the player is moving their own piece
        - If the move stays inside the board
        - If the move doesn't land on their own piece
        - If the move follows the movement rules of that piece
        """

        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]  # Get the piece being moved

        # No piece at the starting position
        if piece == '.':
            print("Invalid move: There's no piece at the selected square.")
            return False

        # Moving an opponent's piece
        if (game_state["turn"] == "white" and piece[0] != 'w') or (game_state["turn"] == "black" and piece[0] != 'b'):
            print("Invalid move: You can only move your own pieces.")
            return False

        # Move is outside the board boundaries
        if not (0 <= end_row < 5 and 0 <= end_col < 5):
            print("Invalid move: Destination is out of bounds.")
            return False

        # Can't land on your own piece
        target_piece = game_state["board"][end_row][end_col]
        if target_piece != '.' and target_piece[0] == piece[0]:
            print("Invalid move: You cannot capture your own piece.")
            return False

        # Calls the correct movement function based on the piece type
        if piece[1] == 'K':
            return self.is_valid_king(start, end)
        elif piece[1] == 'p':
            return self.is_valid_pawn(start, end, piece, game_state)
        elif piece[1] == 'N':
            return self.is_valid_knight(start, end)
        elif piece[1] == 'B':
            return self.is_valid_bishop(start, end, game_state)
        elif piece[1] == 'Q':
            return self.is_valid_queen(start, end, game_state)

        # If somehow it reaches here, move is invalid
        print("Invalid move: Not a recognized piece.")
        return False
    def valid_moves(self, game_state):
        """
        Generates all valid moves for the current player.
        Ensures AI does not attempt to capture its own pieces.
        """
        valid_moves = []
        board = game_state["board"]
        turn = game_state["turn"]
        
        for row in range(5):
            for col in range(5):
                piece = board[row][col]
                if piece == '.' or (turn == 'white' and piece[0] != 'w') or (turn == 'black' and piece[0] != 'b'):
                    continue  # Skip empty squares and opponent's pieces
                
                # Generate possible moves for each piece type
                possible_moves = []
                if piece[1] == 'K':
                    possible_moves = [(row + dr, col + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if dr != 0 or dc != 0]
                elif piece[1] == 'Q':
                    possible_moves = self.generate_straight_moves(row, col, game_state) + self.generate_diagonal_moves(row, col, game_state)
                elif piece[1] == 'B':
                    possible_moves = self.generate_diagonal_moves(row, col, game_state)
                elif piece[1] == 'N':
                    possible_moves = [(row + dr, col + dc) for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]]
                elif piece[1] == 'p':
                    possible_moves = self.generate_pawn_moves(row, col, piece, game_state)
                
                # Filter valid moves (no self-capture, ensure within bounds, no king capture)
                for move in possible_moves:
                    end_row, end_col = move
                    if 0 <= end_row < 5 and 0 <= end_col < 5:  # Ensure move stays within board
                        target_piece = board[end_row][end_col]
                        if target_piece == '.' or (target_piece[0] != piece[0] and target_piece[1] != 'K'):
                            valid_moves.append(((row, col), (end_row, end_col)))
        
        return valid_moves

    def evaluate_board_e1(self, game_state):
        """
        Heuristic e1: Strategic Positional Play
        - Rewards controlling the center.
        - Penalizes repeating positions.
        - Encourages piece mobility.
        """
        print("e1")
        piece_values = {'p': 1, 'B': 3, 'N': 3, 'Q': 9, 'K': 999}
        center_bonus = [[0, 1, 3, 1, 0],
                        [1, 2, 4, 2, 1],
                        [3, 4, 6, 4, 3],
                        [1, 2, 4, 2, 1],
                        [0, 1, 3, 1, 0]]

        score = 0

        for r in range(5):
            for c in range(5):
                piece = game_state["board"][r][c]
                if piece != '.':
                    value = piece_values[piece[1]] + center_bonus[r][c]
                    score += value if piece[0] == 'w' else -value

        # Penalize repeated board states
        board_hash = self.hash_board(game_state)
        repetition_penalty = -10 if self.board_history.get(board_hash, 0) > 1 else 0
        score += repetition_penalty

        # Encourage mobility
        valid_moves = self.valid_moves(game_state)
        score += len(valid_moves) * (1.0 if game_state["turn"] == "white" else -1.0)

        return score

    def evaluate_board_e2(self, game_state):
        """
        Heuristic e2: Aggressive Play
        - Prioritizes capturing opponent’s high-value pieces.
        - Encourages attacking the opponent's King.
        - Encourages open, aggressive play and piece activity.
        - Penalizes passive moves (staying in the same place).
        """

        print("e2")

        piece_values = {'p': 1, 'B': 3, 'N': 3, 'Q': 9, 'K': 999}
        score = 0
        king_positions = {'w': None, 'b': None}

        # Evaluate piece positions
        for r in range(5):
            for c in range(5):
                piece = game_state["board"][r][c]
                if piece != '.':
                    value = piece_values[piece[1]]
                    score += value if piece[0] == 'w' else -value
                    if piece[1] == 'K':
                        king_positions[piece[0]] = (r, c)

        # Increase the weight for capturing opponent’s high-value pieces
        for move in self.valid_moves(game_state):
            _, (end_r, end_c) = move
            target_piece = game_state["board"][end_r][end_c]
            if target_piece != '.':
                capture_value = piece_values.get(target_piece[1], 0)
                score += (capture_value * 4) if game_state["turn"] == 'white' else (-capture_value * 4)

        # King Safety: Penalize exposed kings
        for color, (kr, kc) in king_positions.items():
            if kr is None:
                continue  # King is already captured

            opponent = 'b' if color == 'w' else 'w'
            attack_bonus = sum(
                20 for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= kr + dr < 5 and 0 <= kc + dc < 5 and game_state["board"][kr + dr][kc + dc].startswith(opponent)
            )
            score += attack_bonus if color == 'b' else -attack_bonus

        # Encourage piece mobility
        score += len(self.valid_moves(game_state)) * (1.2 if game_state["turn"] == 'white' else -1.2)

        # Penalize passive moves (repetitive board states)
        board_hash = self.hash_board(game_state)
        repetition_penalty = self.board_history.get(board_hash, 0) * 15
        score -= repetition_penalty if game_state["turn"] == 'white' else -repetition_penalty

        return score

    def generate_straight_moves(self, row, col, game_state):
        """
        Generates valid straight-line moves (Rook-like) for the Queen.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Down, Up, Right, Left
        return self.generate_directional_moves(row, col, game_state, directions)


    def generate_diagonal_moves(self, row, col, game_state):
        """
        Generates valid diagonal moves (Bishop-like) for the Queen and Bishop.
        """
        directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]  # Diagonal directions
        return self.generate_directional_moves(row, col, game_state, directions)


    def generate_directional_moves(self, row, col, game_state, directions):
        """
        Helper function to generate moves in a given set of directions.
        """
        board = game_state["board"]
        moves = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 5 and 0 <= c < 5:
                if board[r][c] == '.':
                    moves.append((r, c))
                elif board[r][c][0] != board[row][col][0]:
                    moves.append((r, c))  # Capture
                    break
                else:
                    break  # Blocked by own piece
                r += dr
                c += dc
        return moves


    def generate_pawn_moves(self, row, col, piece, game_state):
        """
        Generates valid moves for a pawn.
        """
        board = game_state["board"]
        direction = -1 if piece[0] == 'w' else 1  # White moves up, Black moves down
        moves = []
        
        # Normal forward move
        if 0 <= row + direction < 5 and board[row + direction][col] == '.':
            moves.append((row + direction, col))
        
        # Capture diagonally
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5 and board[new_row][new_col] != '.' and board[new_row][new_col][0] != piece[0]:
                moves.append((new_row, new_col))
                
        if (piece == 'wp' and row == 1) or (piece == 'bp' and row == 3):
            return moves

        return moves
    
    #helper fxn that were gonna use to keep track of uncaptured pieces for turns
    def count_pieces(self, game_state):
            """Counts the total number of pieces on the board."""
            return sum(1 for row in game_state["board"] for piece in row if piece != '.')

    def convert_move_to_notation(self, move):
        """
        Converts a move from tuple format ((start_row, start_col), (end_row, end_col))
        to chess notation (e.g., (0, 1), (1, 1) -> 'B1 B2').
        """
        if move == "GAME_OVER":
            return "GAME_OVER"
        
        start, end = move
        start_notation = f"{chr(ord('A') + start[1])}{5 - start[0]}"
        end_notation = f"{chr(ord('A') + end[1])}{5 - end[0]}"
        
        return f"{start_notation} {end_notation}"
    
    def play_ai_vs_ai(self, depth=3, heuristic="e0", max_time=5, max_turns=10, use_alpha_beta=True):
        """
        Runs a game where AI plays against AI until someone wins or a draw occurs.
        Alternates turns between White and Black.
        Displays board state only after AI makes a move (not during Minimax search).
        """
        self.max_turns = max_turns
        print("Starting AI vs AI Game...")
        print(f"→ Heuristic: {heuristic}")
        print(f"→ Alpha-Beta Pruning: {'Enabled' if use_alpha_beta else 'Disabled'}")
        print(f"→ Depth: {depth}, Max Time per Move: {max_time} seconds, Max Turns: {max_turns}\n")
        filename = f"gameTrace-{use_alpha_beta}-{max_time}-{max_turns}.txt"
        with open(filename, "w") as file:
            try:
                file.write(f"Mini Chess AI vs AI Game Trace\n")
                file.write(f"Alpha-Beta: {use_alpha_beta}\nTimeout: {max_time} sec\nMax Turns: {max_turns}\n\n")
                turn_count = 0
                # Set the heuristic evaluation function
                if heuristic == "e1":
                    self.evaluate_board = self.evaluate_board_e1
                elif heuristic == "e2":
                    self.evaluate_board = self.evaluate_board_e2
                else:
                    self.evaluate_board = self.evaluate_board  # Default e0

                while True:
                    # Check if the game has ended
                                        #self.display_board(self.current_game_state)  # Show board before AI move
                    if self.current_game_state["turn"] == "white":
                        turn_count += 1
                    file.write(f"\nTURN {turn_count} ({self.current_game_state['turn'].capitalize()} to move)\n")
                    if turn_count > max_turns:
                        file.write("\nGame ended in a DRAW (Max turns reached).\n")
                        print("Game ended in a draw due to max turn limit.")
                        break

                    game_over, winner = self.is_terminal(self.current_game_state)
                    if game_over:
                        if winner:
                            file.write(f"\nWinner: {winner.capitalize()} wins the game!\n")
                            print(f"AI playing as {winner} wins the game!")
                        else:
                            file.write("\nGame ended in a draw.\n")
                            print("The game has ended in a draw!")
                        break

                    # Check if max_turns has been reached
                    if self.total_half_turns >= max_turns * 2:
                        print(f"Game ended in a draw due to reaching {max_turns} turns.")
                        self.current_game_state["winner"] = "draw"
                        break

                    # Let AI choose move using minimax (with or without alpha-beta)
                    start_time = time.time()
                    move , best_search_score = self.get_ai_move(
                        self.current_game_state,
                        depth=depth,
                        heuristic=heuristic,  # Pass heuristic explicitly
                        max_time=max_time,
                        use_alpha_beta=use_alpha_beta
                    )
                    end_time = time.time()
                    move_time = round(end_time - start_time, 4)
                    if move == "GAME_OVER":
                        file.write("\nNo valid moves left. Game over.\n")
                        print("No valid moves left. Game over.")
                        break  # No valid moves available

                    move_notation = self.convert_move_to_notation(move)
                    file.write(f"AI ({self.current_game_state['turn']}) nodes:{move} Board Move: {move_notation}\n")
                    file.write(f"Alpha-Beta search score: {best_search_score}\n")
                    file.write(f"Time for this action: {move_time} sec\n")
                    # Print the AI's chosen move
                    print(f"AI ({self.current_game_state['turn']}) chose: {self.convert_move_to_notation(move)}")

                    # Apply the move and display new board state
                    self.make_move(self.current_game_state, move)
                    # Get heuristic score after move
                    heuristic_score = self.evaluate_board(self.current_game_state)
                    file.write(f"Heuristic score: {heuristic_score}\n")
                    
                    self.display_board(self.current_game_state)  # Show board after move
                    for row in self.current_game_state["board"]:
                        file.write(' '.join(row) + '\n')
            
                    time.sleep(1)  # Pause briefly for readability
            except Exception as e:
                print(f"Error during game execution: {e}")

        print(f"Game trace saved to: {filename}")        

    def play_ai_game(self, mode, depth, max_time=5, max_turns=10, use_alpha_beta=True):
        """
        Runs a game where AI plays against AI, AI plays against a human, or Human plays against Human.
        Uses heuristic e0 by default.
        """
        print("Starting Game...")
        filename = f"gameTrace-{use_alpha_beta}-{max_time}-{max_turns}.txt"

        with open(filename, "w") as file:
            file.write(f"Mini Chess {mode} Game Trace\n")
            file.write(f"Alpha-Beta: {use_alpha_beta}\nTimeout: {max_time} sec\nMax Turns: {max_turns}\n\n")

            turn_count = 0  # Track the number of full turns (White+Black = 1 turn)
            self.total_half_turns = 0  # Reset half turns counter
            # Ask for heuristics only if AI is involved
            heuristic_choice = "e0"  # Default heuristic
            if mode in ["H-AI", "AI-H"]:
                heuristic_choice = input("Choose heuristic (e0 for basic piece values, e1 for positional advantage, e2 for aggressive captures): ").strip().lower()
                if heuristic_choice not in ["e0", "e1", "e2"]:
                    print("Invalid choice! Defaulting to e0.")
                    heuristic_choice = "e0"
                    

            # Ensure correct heuristic is set
            if heuristic_choice == "e1":
                self.evaluate_board = self.evaluate_board_e1
            elif heuristic_choice == "e2":
                self.evaluate_board = self.evaluate_board_e2
            else:
                self.evaluate_board = self.evaluate_board  # Default to e0

            print(f"Using heuristic: {heuristic_choice}")  # Debugging print
            filename = f"gameTrace-{use_alpha_beta}-{max_time}-{max_turns}.txt"

            # AI moves first if AI-H mode
            if mode == "AI-H":
                self.display_board(self.current_game_state)
                move, best_search_score = self.get_ai_move(self.current_game_state, depth, heuristic=heuristic_choice, max_time=max_time, use_alpha_beta=use_alpha_beta)
                move_notation = self.convert_move_to_notation(move)
                print(f"AI (White) chose: {self.convert_move_to_notation(move)}")
                file.write(f"\nTURN {turn_count + 1} (White to move - AI)\n")
                file.write(f"AI (White) chose: {move_notation}\n")
                file.write(f"Search Score: {best_search_score}\n")
                
    
                self.make_move(self.current_game_state, move)

            while True:
                self.display_board(self.current_game_state)

                # Check if max_turns has been reached
                if self.total_half_turns >= max_turns * 2:
                    print(f"Game ended in a draw due to reaching {max_turns} turns.")
                    file.write("Game ended in a draw due to max turn limit.\n")
                    self.current_game_state["winner"] = "draw"
                    break
                move = None
                turn_count += 1
                file.write(f"\nTURN {turn_count} ({self.current_game_state['turn'].capitalize()} to move)\n")
                if self.current_game_state["turn"] == "white":
                    file.write("White to move\n")
                else:
                    file.write("Black to move\n")
            
                if mode == "H-H":  # Human vs Human (No AI, No Heuristics)
                    move = input(f"{self.current_game_state['turn'].capitalize()} to move (e.g., B2 B3): ")
                    move = self.parse_input(move)
                    if not move or not self.is_valid_move(self.current_game_state, move):
                        print("Invalid move. Try again.")
                        continue
                    move_notation = self.convert_move_to_notation(move)
                    file.write(f"Player ({self.current_game_state['turn']}) chose: {move_notation}\n")

                elif mode == "H-AI":
                    if self.current_game_state["turn"] == "white":  # Human starts in H-AI
                        move = input("Enter your move (e.g., B2 B3): ")
                        move = self.parse_input(move)
                        if not move or not self.is_valid_move(self.current_game_state, move):
                            print("Invalid move. Try again.")
                            continue
                        move_notation = self.convert_move_to_notation(move)
                        file.write(f"Player White (H) chose: {move_notation}\n")
                    else:  # AI moves as Black
                        start_time = time.time()
                        move, best_search_score= self.get_ai_move(self.current_game_state, depth, max_time=max_time, use_alpha_beta=use_alpha_beta)
                        end_time = time.time()
                        move_time = round(end_time - start_time, 4)

                        move_notation = self.convert_move_to_notation(move)
                        file.write(f"AI (Black) chose: {move_notation}\n")
                        file.write(f"Search Score: {best_search_score}\n")
                        file.write(f"Time for this action: {move_time} sec\n")

                        print(f"AI selected move: {move_notation}")

                elif mode == "AI-H":  # AI started first, now human plays
                    if self.current_game_state["turn"] == "black":  # AI is White, human plays Black
                        move = input("Enter your move (e.g., B2 B3): ")
                        move = self.parse_input(move)
                        if not move or not self.is_valid_move(self.current_game_state, move):
                            print("Invalid move. Try again.")
                            continue
                        move_notation = self.convert_move_to_notation(move)
                        file.write(f"Human (black) chose: {move_notation}\n")
                    else:  # AI moves as Black
                            start_time = time.time()
                            move, best_search_score = self.get_ai_move(self.current_game_state, depth, max_time=max_time, use_alpha_beta=use_alpha_beta)
                            end_time = time.time()
                            move_time = round(end_time - start_time, 4)

                            move_notation = self.convert_move_to_notation(move)
                            file.write(f"AI (White) chose: {move_notation}\n")
                            file.write(f"Search Score: {best_search_score}\n")
                            file.write(f"Time for this action: {move_time} sec\n")
                            print(f"AI selected move: {move_notation}")
                self.make_move(self.current_game_state, move)

                               # Log board state after move
                for row in self.current_game_state["board"]:
                        file.write(' '.join(row) + '\n')

                heuristic_score = self.evaluate_board(self.current_game_state)
                file.write(f"Heuristic score: {heuristic_score}\n")

                game_over, winner = self.is_terminal(self.current_game_state)
                if game_over:
                        if winner:
                            print(f"{winner.capitalize()} wins the game!")
                            file.write(f"\nWinner: {winner.capitalize()} wins the game!\n")
                        else:
                            print("The game has ended in a draw!")
                            file.write("\nGame ended in a DRAW.\n")
                        break

    def get_ai_move(self, game_state, depth=3, heuristic="e0", max_time=5, use_alpha_beta=True):
        """
        AI selects the best move within a given time limit.
        - Uses minimax or alpha-beta pruning based on use_alpha_beta.
        - Stops searching if max_time is reached.
        - Prints time taken and nodes explored.
        """
        start_time = time.time()
        self.nodes_explored = 0  # Reset node counter
        self.nodes_explored = 0
        self.depth_stats = {}

        # Ensure correct heuristic is assigned
        if heuristic == "e1":
            self.evaluate_board = self.evaluate_board_e1
        elif heuristic == "e2":
            self.evaluate_board = self.evaluate_board_e2
        else:  # Default to e0
            self.evaluate_board = self.evaluate_board  

        valid_moves = self.valid_moves(game_state)
        if not valid_moves:
            game_state["winner"] = "black" if game_state["turn"] == "white" else "white"
            return "GAME_OVER"

        best_move = None
        best_value = float('-inf') if game_state["turn"] == "white" else float('inf')
        best_search_score = None
        start_time = time.time()

        for move in valid_moves:
            if time.time() - start_time > max_time:
                print(f"AI timeout! Selecting best move so far ({time.time() - start_time:.4f} sec).")
                break

            new_state = copy.deepcopy(game_state)
            self.make_move(new_state, move, simulated=True)

            # Ensure the correct heuristic is used in evaluation
            if use_alpha_beta:
                eval_score, _ = self.minimax(new_state, depth, -math.inf, math.inf, game_state["turn"] == "white")
            else:
                eval_score, _ = self.minimax_without_pruning(new_state, depth, game_state["turn"] == "white")

            if (game_state["turn"] == "white" and eval_score > best_value) or \
            (game_state["turn"] == "black" and eval_score < best_value):
                best_value = eval_score
                best_move = move
                best_search_score = eval_score 

        filename = f"gameTrace-{use_alpha_beta}-{max_time}-{max_turns}.txt"
        with open(filename, "a") as file:
             file.write(f"Alpha-Beta search score: {best_search_score}\n")     
        elapsed_time = time.time() - start_time

        print(f"AI ({game_state['turn']}) took {elapsed_time:.4f} seconds to select a move.")
        print(f"AI explored {self.nodes_explored} nodes using {'Alpha-Beta Pruning' if use_alpha_beta else 'Standard Minimax'}.")
        print(f"Using heuristic: {heuristic}")  # Debugging line to verify heuristic selection
        self.print_cumulative_stats()

        return best_move, best_search_score

    def is_valid_king(self, start, end):
        """
        King moves 1 square in any direction (horizontal, vertical, diagonal).
        If it moves more than 1 square, it's an invalid move.
        """

        row_diff = abs(start[0] - end[0])  # How far the King moves in rows
        col_diff = abs(start[1] - end[1])  # How far the King moves in columns

        return row_diff <= 1 and col_diff <= 1  # Must be within 1 square range

    def is_valid_pawn(self, start, end, piece, game_state):
        """
        Pawns:
        - Move forward 1 square if empty
        - Capture diagonally (1 step) only if an opponent's piece is there
        - Cannot move backwards
        """

        start_row, start_col = start
        end_row, end_col = end
        direction = -1 if piece[0] == 'w' else 1  # White moves UP (-1), Black moves DOWN (+1)

        # Normal forward move (must land on an empty square)
        if start_col == end_col and end_row == start_row + direction and game_state["board"][end_row][end_col] == '.':
            return True

        # Diagonal capture (must be an enemy piece)
        if abs(start_col - end_col) == 1 and end_row == start_row + direction:
            target_piece = game_state["board"][end_row][end_col]
            if target_piece != '.' and target_piece[0] != piece[0]:
                return True  # Capturing opponent’s piece

        print("Invalid move: Pawns can only move forward or capture diagonally.")
        return False  # Invalid move

    def is_valid_knight(self, start, end):
        """
        Knights move in an L-shape:
        - 2 squares in one direction + 1 square in the other
        - Can jump over other pieces
        """

        row_diff = abs(start[0] - end[0])
        col_diff = abs(start[1] - end[1])

        if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
            return True

        print("Invalid move: Knights must move in an L-shape (2-1 or 1-2).")
        return False

    def is_valid_bishop(self, start, end, game_state):
        """
        Bishops move diagonally without jumping over pieces.
        If any piece is in the way, move is invalid.
        """

        row_diff = abs(start[0] - end[0])
        col_diff = abs(start[1] - end[1])

        # Must move diagonally
        if row_diff != col_diff:
            print("Invalid move: Bishops must move diagonally.")
            return False

            # Check for obstacles
        row_step = 1 if end[0] > start[0] else -1
        col_step = 1 if end[1] > start[1] else -1
        r, c = start[0] + row_step, start[1] + col_step

        while (r, c) != end:
            if game_state["board"][r][c] != '.':
                print("Invalid move: Bishops cannot jump over pieces.")
                return False
            r += row_step
            c += col_step

        return True

    def is_valid_queen(self, start, end, game_state):
        """
        The Queen moves like:
        - A Bishop (diagonal movement)
        - Straight in any direction (like a Rook but there's no Rook in this game)
        - Cannot jump over other pieces
        """

        start_row, start_col = start
        end_row, end_col = end

        # Queen moves like a Bishop
        if abs(start_row - end_row) == abs(start_col - end_col):
            return self.is_valid_bishop(start, end, game_state)

        # Queen moves straight like a Rook (Horizontally or Vertically)
        if start_row == end_row or start_col == end_col:
            return self.is_valid_straight_line(start, end, game_state)

        print("Invalid move: Queens must move diagonally or in a straight line.")
        return False

    def is_valid_straight_line(self, start, end, game_state):
        """
        This function checks if a piece moves in a straight line (like a Rook).
        The Queen needs this function since it moves straight in addition to diagonal.
        """

        start_row, start_col = start
        end_row, end_col = end

        # 🔹 Check if the move is in a straight line
        if start_row != end_row and start_col != end_col:
            return False  # Not a straight move

        # 🔹 Check for obstacles
        if start_row == end_row:  # Moving horizontally
            step = 1 if end_col > start_col else -1
            for col in range(start_col + step, end_col, step):
                if game_state["board"][start_row][col] != '.':
                    print("Invalid move: The path is blocked.")
                    return False

        if start_col == end_col:  # Moving vertically
            step = 1 if end_row > start_row else -1
            for row in range(start_row + step, end_row, step):
                if game_state["board"][row][start_col] != '.':
                    print("Invalid move: The path is blocked.")
                    return False

        return True

    def make_move(self, game_state, move, simulated=False):
        """
        Executes a move and updates the board.
        Tracks the number of turns without a capture properly.
        """
        if move == "GAME_OVER":
            return game_state  # Prevents crashing when no moves are available
        # Store piece count before the move - used for tracking piece number before a capture for draw condition
        previous_piece_count = self.count_pieces(game_state)

        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        target_piece = game_state["board"][end_row][end_col]

        # Move the piece
        game_state["board"][end_row][end_col] = piece
        game_state["board"][start_row][start_col] = '.'

        # Check if a king was captured (immediate game end)
        if target_piece in ["wK", "bK"]:
            game_state["winner"] = "white" if target_piece == "bK" else "black"
            if not simulated:
                print(f"Game Over! {game_state['winner'].capitalize()} wins by capturing the King.")
            return game_state
        
        # Check if a piece was captured
        new_piece_count = self.count_pieces(game_state)
        if new_piece_count < previous_piece_count:  # If a piece was removed
            self.total_half_turns = 0  # Reset turn counter

        # No-Capture Turn Tracking
        if target_piece == '.':
            self.no_capture_turns += 1  # Increment no-capture half-turns
        else:
            self.no_capture_turns = 0  # Reset if a piece was captured

      # Promote pawns to Queens when they reach the opposite end of the board
        if piece == 'wp' and end_row == 0:
            game_state["board"][end_row][end_col] = 'wQ'
            print("White pawn promoted to Queen!")
        elif piece == 'bp' and end_row == 4:
            game_state["board"][end_row][end_col] = 'bQ'
            print("Black pawn promoted to Queen!")
        # Switch turns
            # Increase total turn count
        if not simulated:
            self.total_half_turns += 1

        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"

        return game_state
    
    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))

              # Ensure the move stays within the board boundaries
            if not (0 <= start[0] < 5 and 0 <= start[1] < 5 and 0 <= end[0] < 5 and 0 <= end[1] < 5):
                print("Invalid move: Out of bounds.")
                return "INVALID_MOVE"
            return (start, end)
        except:
            print("Invalid input format. Use format like 'B2 B3'.")
            return "INVALID_MOVE"

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """

    def play(self,max_turns=10):
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")

        # Open a game trace file for writing
        with open("game_trace.txt", "w") as file:
            file.write("Mini Chess Game Trace\n")
            file.write(f"Mode: Human vs Human\n\n")

            while True:
                self.display_board(self.current_game_state)

                move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
                if move.lower() == 'exit':
                    print("Game exited.")
                    file.write("Game exited.\n")
                    exit(1)

                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    continue

                # Log the move to the file
                file.write(f"Turn: {self.current_game_state['turn']}\n")
                file.write(f"Move: {move}\n")

                self.make_move(self.current_game_state, move)

                # Log board state
                for row in self.current_game_state["board"]:
                    file.write(' '.join(row) + '\n')
                file.write("\n")

                # Check if a player has won the game by capturing opponent king
                if "winner" in self.current_game_state:
                    # If there a win is detected, logs the result in the game trace file
                    file.write(f"{self.current_game_state['turn'].capitalize()} wins the game!\n")
                    break

    def print_cumulative_stats(self):
        stats_output = []

        stats_output.append("\n AI Cumulative Statistics:")
        stats_output.append(f"a) Cumulative states explored: {self.nodes_explored:,}")

        stats_output.append("b) Cumulative states explored by depth:")
        for d in sorted(self.depth_stats):
            count = self.depth_stats[d]
            stats_output.append(f"   Depth {d}: {count:,}")

        stats_output.append("c) Cumulative % states explored by depth:")
        for d in sorted(self.depth_stats):
            percent = (self.depth_stats[d] / self.nodes_explored) * 100
            stats_output.append(f"   Depth {d}: {percent:.1f}%")

        avg_branching = self.total_branching / self.total_nodes if self.total_nodes > 0 else 0
        stats_output.append(f"d) Average branching factor: {avg_branching:.2f}")

        # Print to terminal
        for line in stats_output:
            print(line)

        # Append to game_trace.txt
        with open("game_trace.txt", "a") as f:
            f.write("\n".join(stats_output))
            f.write("\n")


if __name__ == "__main__":
    game = MiniChess()
    game.board_history = {}  

    # Prompt user for game parameters
    print("Set game parameters:")

    # Time limit per move
    max_time = float(input("Enter maximum allowed time per move (in seconds, default: 5): ").strip() or "5")

    # Maximum turns before game ends in a draw
    max_turns = int(input("Enter maximum number of turns before the game is declared a draw (default: 10): ").strip() or "10")

    # Choose between minimax or alpha-beta pruning
    use_alpha_beta = input("Use Alpha-Beta pruning? (True/False, default: True): ").strip().lower()
    use_alpha_beta = use_alpha_beta in ["true", "yes", "1", ""]

    print("\nSelect game mode:")
    print("1 - AI vs AI")
    print("2 - AI (White) vs Human (Black)")
    print("3 - Human (White) vs AI (Black)")
    print("4 - Human vs Human")

    mode_selection = input("Enter the number corresponding to your choice: ").strip()

    if mode_selection == "1":
        depth = int(input("Enter search depth for AI vs AI (recommended: 6): ").strip() or "6")
        heuristic = input("Choose heuristic (e0 for basic, e1 for positional, e2 for aggressive captures): ").strip().lower() or "e0"
        game.play_ai_vs_ai(depth=depth, heuristic=heuristic, max_time=max_time, max_turns=max_turns, use_alpha_beta=use_alpha_beta)

    elif mode_selection == "2":
        depth = int(input("Enter AI search depth (recommended: 3): ").strip() or "3")
        game.play_ai_game(mode="AI-H", depth=depth, max_time=max_time, max_turns=max_turns, use_alpha_beta=use_alpha_beta)

    elif mode_selection == "3":
        depth = int(input("Enter AI search depth (recommended: 3): ").strip() or "3")
        game.play_ai_game(mode="H-AI", depth=depth, max_time=max_time, max_turns=max_turns, use_alpha_beta=use_alpha_beta)

    elif mode_selection == "4":
        game.play(max_turns=max_turns)  

    else:
        print("Invalid selection. Defaulting to Human vs Human mode.")
        game.play(max_turns=max_turns)

