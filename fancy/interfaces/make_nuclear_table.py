import pickle
import os


def make_nuclear_table(output_path):
    """
    Make nuclear table in which composition type is read from user input.
    Table reads in periodic element and returns (A, Z) of that element.
    Add more elements as needed here.
    """

    table = {
        "p": (1, 1),
    "He": (4, 2),
    "Li": (6, 3),
    "Be": (8, 4),
    "B": (10, 5),
    "C": (12, 6),
    "N": (14, 7),
    "O": (16, 8),
    "F": (18, 9),
    "Ne": (20, 10),
    "Na": (22, 11),
    "Mg": (24, 12),
    "Al": (27, 13),
    "Si": (28, 14),
    "P": (31, 15),
    "S": (32, 16),
    "Cl": (35, 17),
    "Ar": (40, 18),
    "K": (40, 19),
    "Ca": (40, 20),
    "Sc": (45, 21),
    "Ti": (48, 22),
    "V": (51, 23),
    "Cr": (52, 24),
    "Mn": (55, 25),
    "Fe": (56, 26),
    }

    with open(os.path.join(output_path, "nuclear_table.pkl"), "wb") as f:

        pickle.dump(table, f)


if __name__ == "__main__":
    this_fpath = os.path.abspath(os.path.dirname(__file__))

    table = {
        "p": (1, 1),
    "He": (4, 2),
    "Li": (6, 3),
    "Be": (8, 4),
    "B": (10, 5),
    "C": (12, 6),
    "N": (14, 7),
    "O": (16, 8),
    "F": (18, 9),
    "Ne": (20, 10),
    "Na": (22, 11),
    "Mg": (24, 12),
    "Al": (27, 13),
    "Si": (28, 14),
    "P": (31, 15),
    "S": (32, 16),
    "Cl": (35, 17),
    "Ar": (40, 18),
    "K": (40, 19),
    "Ca": (40, 20),
    "Sc": (45, 21),
    "Ti": (48, 22),
    "V": (51, 23),
    "Cr": (52, 24),
    "Mn": (55, 25),
    "Fe": (56, 26),
    }

    with open(os.path.join(this_fpath, "nuclear_table.pkl"), "wb") as f:
        pickle.dump(table, f)
