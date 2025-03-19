import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()
        self.no_capture_turns = 0  # Track turns without a capture

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

    def is_terminal(self, game_state):
        """
        Checks if the game is in a terminal state.
        Fix: Ensures the game ends only when a valid win/draw condition occurs.
        """
        # Check if any King is missing
        kings = {piece for row in game_state["board"] for piece in row if piece in ("wK", "bK")}
        if "wK" not in kings:
            game_state["winner"] = "black"
            return True  # Black wins
        if "bK" not in kings:
            game_state["winner"] = "white"
            return True  # White wins
        
        # Check for no valid moves (stalemate/loss condition)
        if not self.valid_moves(game_state):
            game_state["winner"] = "draw"
            return True  # No valid moves means the game is over
        
        # Check draw condition
        if self.no_capture_turns >= 20: #10 full turns
            game_state["winner"] = "draw"
            return True  # Draw
        
        return False

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


    def minimax(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta pruning.
        """
        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate_board(game_state), None

        valid_moves = self.valid_moves(game_state)
        if not valid_moves:  # If no valid moves exist, return game over
            print("DEBUG: Minimax found no valid moves. Returning game over condition.")
            return float('-inf') if maximizing_player else float('inf'), None

        best_move = None

        if maximizing_player:  # White (Maximizing)
            max_eval = -math.inf
            for move in valid_moves:
                new_state = copy.deepcopy(game_state)
                self.make_move(new_state, move, simulated=True)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:  # Prune
                    break
            return max_eval, best_move

        else:  # Black (Minimizing)
            min_eval = math.inf
            for move in valid_moves:
                new_state = copy.deepcopy(game_state)
                self.make_move(new_state, move, simulated=True)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:  # Prune
                    break
            return min_eval, best_move


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
        Heuristic e1: Positional strategy.
        - Encourages control of the center squares.
        - Pieces in center squares get extra weight.
        """
        piece_values = {'p': 1, 'B': 3, 'N': 3, 'Q': 9, 'K': 999}
        center_bonus = [[0, 1, 2, 1, 0],
                        [1, 2, 3, 2, 1],
                        [2, 3, 4, 3, 2],
                        [1, 2, 3, 2, 1],
                        [0, 1, 2, 1, 0]]

        score = 0
        for r in range(5):
            for c in range(5):
                piece = game_state["board"][r][c]
                if piece != '.':
                    value = piece_values[piece[1]] + center_bonus[r][c]
                    score += value if piece[0] == 'w' else -value
        return score

    def evaluate_board_e2(self, game_state):
        """
        Heuristic e2: Aggressive strategy.
        - Focuses on capturing opponent’s pieces.
        - Extra bonus if opponent's King is trapped.
        """
        piece_values = {'p': 1, 'B': 3, 'N': 3, 'Q': 9, 'K': 999}
        score = 0

        for r in range(5):
            for c in range(5):
                piece = game_state["board"][r][c]
                if piece != '.':
                    value = piece_values[piece[1]]
                    score += value if piece[0] == 'w' else -value

        # Bonus if opponent’s King is trapped
        for king, opponent in [('wK', 'b'), ('bK', 'w')]:
            king_pos = [(r, c) for r in range(5) for c in range(5) if game_state['board'][r][c] == king]
            if king_pos:
                r, c = king_pos[0]
                moves = [(r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if
                         0 <= r + dr < 5 and 0 <= c + dc < 5]
                if all(game_state['board'][mr][mc] != '.' and game_state['board'][mr][mc][0] == opponent for mr, mc in
                       moves):
                    score += 50 if king == 'bK' else -50  # Bonus if opponent’s King is trapped

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
    
    def get_ai_move(self, game_state, depth=3, heuristic="e1"):
        """
        Gets the best AI move using minimax algorithm with alpha-beta pruning.
        Ensures AI does not end the game incorrectly.
        """
        if heuristic == "e1":
            self.evaluate_board = self.evaluate_board_e1
        else:
            self.evaluate_board = self.evaluate_board_e2

        # Get valid moves
        valid_moves = self.valid_moves(game_state)
        if not valid_moves:
            print("DEBUG: No valid moves available for AI.")
            game_state["winner"] = "black" if game_state["turn"] == "white" else "white"
            return "GAME_OVER"  # AI has no legal moves, game ends.

        _, best_move = self.minimax(game_state, depth, -math.inf, math.inf, game_state["turn"] == "white")

        if best_move is None:
        
            return valid_moves[0]  # Pick a random valid move instead of ending the game
        
      
        return best_move
    
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
    
    def play_ai_game(self, mode="AI-AI", depth=3):
        """
        Runs a game where AI plays against AI, AI plays against a human, or Human plays against Human.
        Uses heuristic e0 by default. If AI is involved, user can choose e1 or e2.
        """
        print("Starting Game...")

        # Default heuristic: e0
        self.evaluate_board = self.evaluate_board  # Keep e0 as default

        # Ask for heuristics only if AI is involved
        if mode in ["AI-AI", "H-AI"]:
            heuristic_choice = input("Choose heuristic (e0 for basic piece values, e1 for positional advantage, e2 for aggressive captures): ").strip().lower()
            if heuristic_choice == "e1":
                self.evaluate_board = self.evaluate_board_e1
            elif heuristic_choice == "e2":
                self.evaluate_board = self.evaluate_board_e2
            elif heuristic_choice != "e0":
                print("Invalid choice! Defaulting to e0.")

        while True:
            self.display_board(self.current_game_state)

            if mode == "H-H":  # Human vs Human (No AI, No Heuristics)
                move = input(f"{self.current_game_state['turn'].capitalize()} to move (e.g., B2 B3): ")
                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    continue

            elif mode == "H-AI" and self.current_game_state["turn"] == "white":
                move = input("Enter your move (e.g., B2 B3): ")
                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    continue
            else:  # AI Move (For AI-AI or AI playing in H-AI)
                move = self.get_ai_move(self.current_game_state, depth)
                print(f"AI selected move: {self.convert_move_to_notation(move)}")

            self.make_move(self.current_game_state, move)

            if "winner" in self.current_game_state:
                if self.current_game_state["winner"] == "draw":
                    print("The game has reached a draw. Game over.")
                else:
                    print(f"{self.current_game_state['winner'].capitalize()} wins the game!")
                break

            
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

        # 🔹 Queen moves like a Bishop
        if abs(start_row - end_row) == abs(start_col - end_col):
            return self.is_valid_bishop(start, end, game_state)

        # 🔹 Queen moves straight like a Rook (Horizontally or Vertically)
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
        - Ensures AI does not make illegal self-capturing moves.
        - Corrects pawn promotion so it only prints when a real move is played.
        - The `simulated` flag prevents promotions from being printed during Minimax search.
        """
        if move == "GAME_OVER":
            return game_state  # Prevents crashing when no moves are available

        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]

        # Ensure move stays within board boundaries
        if not (0 <= end_row < 5 and 0 <= end_col < 5):
            if not simulated:
                print("Error: Move out of board boundaries.")
            return game_state  # Ignore the move

        target_piece = game_state["board"][end_row][end_col]

        # Prevent self-capture (should not happen with valid_moves, but double-check)
        if target_piece != '.' and target_piece[0] == piece[0]:
            if not simulated:
                print("Error: AI attempted an invalid self-capturing move.")
            return game_state  # Do not execute the move

        # Move the piece
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece

        # Check if a king was captured
        if target_piece in ["wK", "bK"]:
            game_state["winner"] = "white" if target_piece == "bK" else "black"
            if not simulated:
                print(f"Game Over! {game_state['winner'].capitalize()} wins by capturing the King.")
            return game_state

        # **Pawn Promotion Logic** (Fix: Only print if it's not a simulation)
        if piece == 'wp' and end_row == 0:
            game_state["board"][end_row][end_col] = 'wQ'
            if not simulated:
                print("White pawn promoted to Queen!")
        elif piece == 'bp' and end_row == 4:
            game_state["board"][end_row][end_col] = 'bQ'
            if not simulated:
                print("Black pawn promoted to Queen!")

        # Track turns with no captures
        if target_piece == '.':
            self.no_capture_turns += 1
        else:
            self.no_capture_turns = 0  # Reset if a piece was captured

        # Switch turns
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

    def play(self):
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
                if 'wins' in self.current_game_state:
                    # If there a win is detected, logs the result in the game trace file
                    file.write(f"{self.current_game_state['turn'].capitalize()} wins the game!\n")
                    break


if __name__ == "__main__":
    game = MiniChess()
    game.play_ai_game(mode="H-AI", depth=3)
