from collections.abc import Iterator
from enum import Enum
from typing import Literal, NamedTuple, Optional

class Piece(object):
    def __str__(self) -> str:
        ...
    def __init__(self, color):
        self.color = color

    def move(self, start: str, end: str, board):
        ...

    def is_white(self) -> bool:
        return True if self.color == "white" else False


class Pawn(Piece):
    def __str__(self) -> str:
        return "P" if self.color == "white" else "p"

    def verify_pawn(self, start: str, end: str, board) -> bool:
        fig = board.board[start]
        v = get_vector(start=start, end=end, board = board)

        """Пешка."""
        if not fig.is_white() and start[-1] == "7" and v == (0, -2):
            return True
        if fig.is_white() and start[-1] == "2" and v == (0, 2):
            return True

        if not fig.is_white() and (v == (-1, -1) or v == (1, -1)) and end in board.board:
            return True
        if fig.is_white() and (v == (1, 1) or v == (-1, 1)) and end in board.board:
            return True

        if not fig.is_white() and v == (0, -1):
            return True
        if fig.is_white() and v == (0, 1):
            return True

        return False


class Rook(Piece):
    def __str__(self) -> str:
        return "R" if self.color == "white" else "r"

    def verify_rook(self, vector: tuple[int, int]) -> bool:
        """Ладья."""
        return vector[0] == 0 or vector[1] == 0

class Knight(Piece):
    def __str__(self) -> str:
        return "N" if self.color == "white" else "n"

    def verify_knight(self, vector: tuple[int, int]) -> bool:
        """Конь."""
        vector_vals = {abs(v) for v in vector}
        return vector_vals == {1, 2}

class Bishop(Piece):
    def __str__(self) -> str:
        return "B" if self.color == "white" else "b"

    def verify_bishop(self, vector: tuple[int, int]) -> bool:
        """Слон."""
        return abs(vector[0]) == abs(vector[1])

class Queen(Piece):
    def __str__(self) -> str:
        return "Q" if self.color == "white" else "q"

    def verify_queen(self, vector: tuple[int, int]) -> bool:
        """Королева."""
        x, y = vector
        return (abs(x) == abs(y)) or x == 0 or y == 0

class King(Piece):
    def __str__(self) -> str:
        return "K" if self.color == "white" else "k"

    def verify_king(self, vector: tuple[int, int]) -> bool:
        """Король."""
        return abs(vector[0]) <= 1 and abs(vector[1]) <= 1

class Board(object):
    COLUMN_NAMES = "ABCDEFGH"
    SIZE_X = SIZE_Y = 8
    board = {}
    p = Pawn("black")
    r = Rook("black")
    n = Knight("black")
    b = Bishop("black")
    q = Queen("black")
    k = King("black")

    P = Pawn("white")
    R = Rook("white")
    N = Knight("white")
    B = Bishop("white")
    Q = Queen("white")
    K = King("white")

    for column in range(SIZE_Y):
        board[COLUMN_NAMES[column] + "8"] = [r, n, b, q, k, b, n, r][column]
        board[COLUMN_NAMES[column] + "7"] = p
        board[COLUMN_NAMES[column] + "1"] = [R, N, B, Q, K, B, N, R][column]
        board[COLUMN_NAMES[column] + "2"] = P

    def __str__(self) -> str:
        string = "\n   A B C D E F G H\n"
        for row_i in reversed(range(1, self.SIZE_Y + 1)):
            string += str(row_i) + "  "
            for column in self.COLUMN_NAMES:
                pos = f"{column}{row_i}"
                string += self.board[pos].__str__() if pos in self.board else "."
                string += " "
            string += " " + str(row_i) + "\n"
        string += "\n   A B C D E F G H\n"
        return string

    def num_white_pieces(self) -> int:
        count = 0
        for i in self.board.values():
            if i.is_white():
                count += 1
        return count

    def num_black_pieces(self) -> int:
        count = 0
        for i in self.board.values():
            if not i.is_white():
                count += 1
        return count

    def balance(self) -> int:
        count = 0
        for i in self.board.values():
            if i.__class__ == Pawn:
                if i.is_white():
                    count += 1
                else:
                    count -= 1
            elif i.__class__ == Knight or i.__class__ == Bishop:
                if i.is_white():
                    count += 3
                else:
                    count -= 3
            elif i.__class__ == Rook:
                if i.is_white():
                    count += 5
                else:
                    count -= 5
            elif i.__class__ == Queen:
                if i.is_white():
                    count += 9
                else:
                    count -= 9

        return count

    def __contains__(self, piece: Piece) -> bool:
        for i in self.board.values():
            if i.__class__ == piece:
               return True

        return False

    def __getitem__(self, pos) -> Piece:
        return self.board[pos]



GameState = NamedTuple(
    "GameState",
    [
        ("board", Board),
        ("player_counters", dict[str, int]),
        ("is_player_white", bool),
    ],
)

ErrorType = Enum(
    "ErrorType",
    {
        "FORMAT": "Wrong input format",
        "INVALID_MOVE": "The piece cannot make the specified move",
    },
)




def is_fig_friendly(fig: Piece, state: GameState) -> bool:
    return fig.is_white() == state.is_player_white

def is_incorrect_pos(pos: str, board: Board) -> bool:
    return (
        not len(pos) == 2
        or pos[-1] not in "12345678"
        or pos[0] not in board.COLUMN_NAMES
    )


def pos_to_int(pos: str, board: Board) -> tuple[int, int]:
    x, y = tuple(pos)
    return board.COLUMN_NAMES.index(x), int(y)


def pos_from_int(pos_int: tuple[int, int], board: Board) -> str:
    x, y = pos_int
    return f"{board.COLUMN_NAMES[x]}{y}"


def get_vector(start: str, end: str, board: Board) -> tuple[int, int]:
    """Функция, возвращающая вектор перемещения фигуры."""
    start_v, end_v = (
        (board.COLUMN_NAMES.index(start[0]), int(start[-1])),
        (board.COLUMN_NAMES.index(end[0]), int(end[-1])),
    )
    return end_v[0] - start_v[0], end_v[-1] - start_v[-1]


def sgn(number: int) -> Literal[-1, 0, 1]:
    """Вспомогательная функция, определяющая знак переданного числа."""
    if number > 0:
        return 1
    if number < 0:
        return -1
    return 0


def get_straight_path(start: str, end: str, board: Board) -> Iterator[Optional[str]]:
    v = get_vector(start, end, board)
    is_vector_straight = 0 in v

    if not is_vector_straight or start == end:
        return None

    start_int = pos_to_int(start, board)
    end_int = pos_to_int(end, board)

    zero_dim = v.index(0)
    non_zero_dim = 1 - zero_dim

    start_i, end_i = start_int[non_zero_dim], end_int[non_zero_dim]
    if start_i > end_i:
        start_i, end_i = end_i, start_i

    for i in range(start_i + 1, end_i):
        new_pos_int_ = list(start_int)
        new_pos_int_[non_zero_dim] = i
        new_pos = pos_from_int(tuple(new_pos_int_), board)  # type: ignore[arg-type]
        yield new_pos


def get_diag_path(start: str, end: str, board: Board) -> Iterator[Optional[str]]:
    v = get_vector(start, end, board)
    start_int = pos_to_int(start, board)
    end_int = pos_to_int(end, board)

    is_vector_diag = abs(v[0]) == abs(v[1])
    if not is_vector_diag or start == end:
        return None

    x_sgn = sgn(v[0])
    y_sgn = sgn(v[1])
    x_values = range(start_int[0] + x_sgn, end_int[0], x_sgn)
    y_values = range(start_int[1] + y_sgn, end_int[1], y_sgn)
    for i, j in zip(x_values, y_values):
        new_pos_int = (i, j)
        new_pos = pos_from_int(new_pos_int, board)
        yield new_pos


def has_obstacle(start: str, end: str, board: Board) -> bool:
    """Функция проверяет наличие препятствий по ходу фигуры."""
    for path_func in (get_straight_path, get_diag_path):
        path = path_func(start, end, board)
        for temp_pos in path:
            if temp_pos is None:
                break

            if temp_pos in board.board:
                return True

    return False

def switch_player(state: GameState) -> GameState:
    args = state._asdict()
    args["is_player_white"] = not args["is_player_white"]
    return GameState(**args)


def update_player_counter(state: GameState) -> None:
    name = get_player_name(state.is_player_white)
    state.player_counters[name] += 1


def get_player_name(is_player_white: bool) -> str:
    return "white" if is_player_white else "black"


def is_cmd_exit(inp: str) -> bool:
    return inp == "exit"


def log_error(err: ErrorType) -> None:
    print(f"Error. Type: {err.value}.")


def init_state() -> GameState:
    board = Board

    return GameState(
        board=board,
        player_counters={
            "white": 0,
            "black": 0,
        },
        is_player_white=True,
    )



def try_draw(inp: str, board: Board) -> bool:
    if inp == "draw":
        draw_board(board)
        return True
    return False

def try_balance(inp: str, board: Board) -> bool:
    if inp.split()[0] == "balance" and (inp.split()[1] == "black" or inp.split()[1] == "white"):
        if inp.split()[1] == "white":
            print(board.balance(board))
        else:
            print(-board.balance(board))
        return True
    return False

def try_on_board(inp: str, board: Board) -> bool:
    arr = inp.split()
    if len(arr) == 3:
        if arr[1] == "on" and arr[2] == "board":
            if arr[0] == "queen":
                print(board.__contains__(board, Queen))
                return True
            elif arr[0] == "pawn":
                print(board.__contains__(board, Pawn))
                return True
            elif arr[0] == "knight":
                print(board.__contains__(board, Knight))
                return True
            elif arr[0] == "king":
                print(board.__contains__(board, King))
                return True
            elif arr[0] == "bishop":
                print(board.__contains__(board, Bishop))
                return True
            elif arr[0] == "rook":
                print(board.__contains__(board, Rook))
                return True
        else:
            False
    else:
        False



def draw_board(board: Board) -> None:
    print(board.__str__(board))

def parse_move(inp: str, board: Board) -> tuple[Optional[tuple[str, str]], Optional[ErrorType]]:
    """Данная функция обрабатывает ввод хода: производит несколько проверок корректности и вызывает ранее объявленные функции проверок."""

    step = inp.upper().split("-")
    if len(step) != 2:
        return None, ErrorType.FORMAT
    pos_start, pos_end = step[0], step[1]
    if pos_start == pos_end:
        return None, ErrorType.INVALID_MOVE
    if is_incorrect_pos(pos_start, board) or is_incorrect_pos(pos_end, board):
        return None, ErrorType.FORMAT

    return ((pos_start, pos_end), None)


def move_fig(start: str, end: str, state: GameState) -> None:
    fig = state.board.board.pop(start)
    state.board.board[end] = fig


def validate_move(
    start: str, end: str, state: GameState
) -> tuple[bool, Optional[ErrorType]]:
    fig = None if start not in state.board.board.keys() else state.board.board[start]
    if fig is None:
        return False, ErrorType.INVALID_MOVE

    if not is_fig_friendly(fig, state=state):
        return False, ErrorType.INVALID_MOVE

    if not validate_trajectory(start=start, end=end, board=state.board):
        return False, ErrorType.INVALID_MOVE

    is_knight = fig.__class__ == Knight
    if not is_knight and has_obstacle(start=start, end=end, board=state.board):
        return False, ErrorType.INVALID_MOVE

    is_pawn = fig.__class__ == Pawn
    is_end_empty = end not in state.board.board
    end_fig = None if is_end_empty else state.board.board[end]

    is_end_enemy = end_fig is not None and not is_fig_friendly(
        end_fig, state=state
    )

    v = get_vector(start, end, state.board)
    is_move_diag = v[0] != 0

    is_valid_pawn_move = is_end_empty or (is_move_diag and is_end_enemy)

    if is_pawn and not is_valid_pawn_move:
        return False, ErrorType.INVALID_MOVE

    if end in state.board.board:
        if is_fig_friendly(state.board.board[end], state=state):
            return False, ErrorType.INVALID_MOVE

    return True, None


def validate_trajectory(start: str, end: str, board: Board) -> bool:
    """Проверяет, может ли выбранная фигура ходить по заданной игроком траектории."""
    fig = board.board[start]
    v = get_vector(start, end, board)

    if fig.__class__ == Pawn:
        return Pawn.verify_pawn(fig, start, end, board)
    if fig.__class__ == Knight:
        return Knight.verify_knight(fig, v)
    if fig.__class__ == Bishop:
        return Bishop.verify_bishop(fig, v)
    if fig.__class__ == Queen:
        return Queen.verify_queen(fig, v)
    if fig.__class__ == King:
        return King.verify_king(fig, v)
    if fig.__class__ == Rook:
        return Rook.verify_rook(fig, v)
    return False

def main() -> None:
    state = init_state()

    while True:
        player_name = get_player_name(state.is_player_white)
        player_step_num = state.player_counters[player_name] + 1
        cmd = input(f"{player_name} {player_step_num}:\n")

        if try_draw(cmd, board=state.board):
            continue

        if is_cmd_exit(cmd):
            return

        if try_balance(cmd, board=state.board):
            continue

        if try_on_board(cmd, board=state.board):
            continue

        move, err = parse_move(cmd, state.board)
        if err is not None:
            log_error(err)
            continue
        assert move is not None
        pos_start, pos_end = move

        _, err = validate_move(
            start=pos_start,
            end=pos_end,
            state=state,
        )
        if err is not None:
            log_error(err)
            continue

        move_fig(start=pos_start, end=pos_end, state=state)

        update_player_counter(state)
        state = switch_player(state)


if __name__ == "__main__":
    main()

