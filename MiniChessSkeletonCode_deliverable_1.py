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

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
    """
    def valid_moves(self, game_state):
        # Return a list of all the valid moves.
        # Implement basic move validation
        # Check for out-of-bounds, correct turn, move legality, etc
        return

    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """

    def make_move(self, game_state, move):
        """
        Executes a move and updates the board.
        - Checks if a King was captured (Win Condition).
        - Tracks turns without a capture (Draw Condition).
        """

        start, end = move
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        target_piece = game_state["board"][end_row][end_col]  # Check if capturing a piece

        # Move the piece
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece

        # Check if a King was captured (Win Condition)
        if target_piece == 'bK':
            print("White wins! The Black King has been captured.")
            #sets the winner of the game to 'white' if black king is captured
            self.current_game_state['winner'] = 'white'
            exit(0)

        elif target_piece == 'wK':
            print("Black wins! The White King has been captured.")
            #sets the winner of the game to black if the white king is captured
            self.current_game_state['winner'] = 'black'
            exit(0)

        # Track turns with no captures
        if target_piece == '.':
            self.no_capture_turns += 1
        else:
            self.no_capture_turns = 0  # Reset count if a piece is captured

        # If 10 full turns (20 moves) pass with no captures, declare a draw
        if self.no_capture_turns >= 20:  # 10 turns = 20 moves (White+Black)
            print("Draw! No pieces have been captured in the last 10 turns.")
            exit(0)

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
            return (start, end)
        except:
            return None

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
    game.play()