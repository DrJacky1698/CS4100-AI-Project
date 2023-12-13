import chess
import random
import pandas as pd

class ChessGameToFEN:
    def __init__(self, moves):
        self.moves = moves.split()
        self.board = chess.Board()

    def process_moves(self, randomize=True):
        total_moves = len(self.moves)

        # skip this game, there is empty moves be done
        if total_moves == 0:
            return None

        # randomly select of which game moment to pick
        # there is situation in which game end in one step(idk but it happen while running)
        # this check is to avoid error from random.randint because it do not accept
        # soemthing like random.randit(1,1)
        if total_moves == 1:
            moves_to_process = 1
        else:
            # randomly select of which game moment to be picked
            moves_to_process = random.randint(1, total_moves) if randomize else total_moves

        for move in self.moves[:moves_to_process]:
           if not self.apply_move(move):
                return None  # invalid move found, abort this game

        return self.board.fen()

    def apply_move(self, move):
        try:
            # Extract the move part from "W1.d4" -> "d4"
            move_obj = self.board.parse_san(move.split('.')[1])
            self.board.push(move_obj)
            return True
        except ValueError:
            print(f"Invalid move: {move}")
            return False

    @classmethod
    def read_games_and_convert_to_fen(cls, file_path, output_file, randomize=True, sample_fraction=0.1, chunk_size=100):
        with open(file_path, 'r') as file:
            total_games = sum(1 for line in file if not line.startswith('#') and '###' in line)

        selected_game_indices = random.sample(range(total_games), int(total_games * sample_fraction))
        current_game_index = -1
        chunk_data = []
        games_record_saved = 0

        with open(file_path, 'r') as file:
            for line in file:
                if not line.startswith('#') and '###' in line:
                    current_game_index += 1
                    if current_game_index in selected_game_indices:
                        moves_part = line.split('### ')[-1]
                        chess_game = cls(moves_part)
                        game_fen = chess_game.process_moves(randomize=randomize)
                        if game_fen is not None:  # only add valid games
                            chunk_data.append([current_game_index + 1, game_fen])
                            games_record_saved += 1

                            if len(chunk_data) == chunk_size:
                                cls.save_chunk_to_csv(chunk_data, output_file)
                                print(f"Saved {games_record_saved} games so far.")
                                chunk_data.clear()  # Clear the array for the next 100 data

        # remaining data when read the end of file (probabilty not used)
        if chunk_data:
            cls.save_chunk_to_csv(chunk_data, output_file)

    @staticmethod
    def save_chunk_to_csv(chunk_data, output_file):
        df = pd.DataFrame(chunk_data, columns=["Game Number", "FEN"])
        df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

file_path = 'Util/chessdata1.txt'
output_file = 'Util/chess_games_fen_output.csv'
ChessGameToFEN.read_games_and_convert_to_fen(file_path, output_file)

