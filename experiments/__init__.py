DISPLAY_NAME = {
    # Algorithms
    "dqn": "DQN",
    "adadqnstatic": "AdaDQN",
    "adadqn": "AdaDQN",
    "rsdqn": "RS DQN",
    "dehbdqn": "DEHB DQN",
    # Environments
    "lunar_lander": "Lunar Lander",
    "atari": "Atari",
    "craftax": "Craftax",
}

AVAILABLE_COLORS = {
    "black": "#000000",
    "blue": "#1F77B4",
    "light_blue": "#AEC7E8",
    "orange": "#FF7F0E",
    "light_orange": "#FFBB78",
    "green": "#2CA02C",
    "light_green": "#98DF8A",
    "red": "#D1797A",
    "light_red": "#FF9896",
    "purple": "#9467BD",
    "light_purple": "#C5B0D5",
    "brown": "#8C564B",
    "light_brown": "#C49C94",
    "pink": "#E377C2",
    "light_pink": "#F7B6D2",
    "grey": "#7F7F7F",
    "light_grey": "#C7C7C7",
    "yellow": "#DEDE00",
    "light_yellow": "#F0E886",
    "cyan": "#17BECF",
    "light_cyan": "#9EDAE5",
}


COLORS = {
    "adadqn": AVAILABLE_COLORS["green"],
    "rsdqn": AVAILABLE_COLORS["orange"],
    "dehbdqn": AVAILABLE_COLORS["blue"],
}

ORDERS = {
    "adadqn": 5,
    "rsdqn": 3,
    "dehbdqn": 4,
}
