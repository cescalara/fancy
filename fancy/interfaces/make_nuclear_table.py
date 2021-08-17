import pickle
import os

'''Make nuclear table in which composition type is read from user input.
Table reads in periodic element and returns (A, Z) of that element.
Add more elements as needed here.'''


if __name__ == "__main__":
    this_fpath = os.path.abspath(os.path.dirname(__file__))

    table = {
        "p" : (1, 1),
        "H" : (1, 1),
        "He" : (4, 2),
        "Li" : (7, 3),
        "C" : (12,6),
        "N" : (14, 7),
        "O" : (16, 8),
        "Si" : (28, 14),
        "Fe" : (56, 26)
    }

    with open(os.path.join(this_fpath, "nuclear_table.pkl"), "wb") as f:
        pickle.dump(table, f)