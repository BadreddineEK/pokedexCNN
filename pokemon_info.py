# Métadonnées des 10 Pokémon sélectionnés pour le projet PokéMAM
#
# ⚠️  Ces 10 Pokémon correspondent EXACTEMENT aux classes du modèle entraîné.
#     L'ordre est ALPHABÉTIQUE car Keras (flow_from_directory) trie les dossiers
#     automatiquement. Ne pas modifier cet ordre.
#
#   Index | Pokémon
#   ------|----------
#     0   | blastoise
#     1   | charizard
#     2   | eevee
#     3   | gardevoir
#     4   | gengar
#     5   | gyarados
#     6   | lucario
#     7   | mewtwo
#     8   | pikachu
#     9   | snorlax

POKEMON_CLASSES = [
    "blastoise",   # 0
    "charizard",   # 1
    "eevee",       # 2
    "gardevoir",   # 3
    "gengar",      # 4
    "gyarados",    # 5
    "lucario",     # 6
    "mewtwo",      # 7
    "pikachu",     # 8
    "snorlax",     # 9
]

POKEMON_INFO = {
    "blastoise": {
        "numero": "#009",
        "emoji": "🐢",
        "types": ["Water"],
        "description": "Blastoise has water cannons inside its shell. It crushes its foe under its heavy body and fires jets of water with precision.",
        "hp": 79, "atk": 83, "def": 100, "spa": 85, "spd": 105, "spe": 78,
        "legendaire": False,
    },
    "charizard": {
        "numero": "#006",
        "emoji": "🔥",
        "types": ["Fire", "Flying"],
        "description": "Charizard flies around the sky in search of powerful opponents. It breathes fire so hot it can melt glaciers.",
        "hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100,
        "legendaire": False,
    },
    "eevee": {
        "numero": "#133",
        "emoji": "🦊",
        "types": ["Normal"],
        "description": "Eevee has an unstable genetic makeup that allows it to evolve into 8 different forms depending on its environment.",
        "hp": 55, "atk": 55, "def": 50, "spa": 45, "spd": 65, "spe": 55,
        "legendaire": False,
    },
    "gardevoir": {
        "numero": "#282",
        "emoji": "🤍",
        "types": ["Psychic", "Fairy"],
        "description": "Gardevoir can predict the future and will give its life to protect its Trainer. It can distort gravity around itself.",
        "hp": 68, "atk": 65, "def": 65, "spa": 125, "spd": 115, "spe": 80,
        "legendaire": False,
    },
    "gengar": {
        "numero": "#094",
        "emoji": "👻",
        "types": ["Ghost", "Poison"],
        "description": "Gengar hides in the shadow of others and waits for a chance to steal their heat. It delights in frightening people.",
        "hp": 60, "atk": 65, "def": 60, "spa": 130, "spd": 75, "spe": 110,
        "legendaire": False,
    },
    "gyarados": {
        "numero": "#130",
        "emoji": "🐉",
        "types": ["Water", "Flying"],
        "description": "Gyarados is infamous for its terrifying power. Once it begins to rampage, it won't stop until everything is destroyed.",
        "hp": 95, "atk": 125, "def": 79, "spa": 60, "spd": 100, "spe": 81,
        "legendaire": False,
    },
    "lucario": {
        "numero": "#448",
        "emoji": "🐾",
        "types": ["Fighting", "Steel"],
        "description": "Lucario reads the aura of all living things through its special sense. It can identify and take in the feelings of others.",
        "hp": 70, "atk": 110, "def": 70, "spa": 115, "spd": 70, "spe": 90,
        "legendaire": False,
    },
    "mewtwo": {
        "numero": "#150",
        "emoji": "🔮",
        "types": ["Psychic"],
        "description": "Created by genetic manipulation from Mew, Mewtwo is the most powerful Pokemon ever designed. Its heart is cold.",
        "hp": 106, "atk": 110, "def": 90, "spa": 154, "spd": 90, "spe": 130,
        "legendaire": True,
    },
    "pikachu": {
        "numero": "#025",
        "emoji": "⚡",
        "types": ["Electric"],
        "description": "Pikachu stores electricity in its cheeks and releases it in powerful bursts. The mascot of the entire Pokemon franchise.",
        "hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90,
        "legendaire": False,
    },
    "snorlax": {
        "numero": "#143",
        "emoji": "😴",
        "types": ["Normal"],
        "description": "Snorlax eats 400 kg of food per day and then falls asleep immediately. Nothing can wake it once it is asleep.",
        "hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30,
        "legendaire": False,
    },
}

# Couleurs associées aux types (pour les graphiques)
TYPE_COLORS = {
    "Normal":   "#A8A878",
    "Fire":     "#F08030",
    "Water":    "#6890F0",
    "Electric": "#F8D030",
    "Grass":    "#78C850",
    "Ice":      "#98D8D8",
    "Fighting": "#C03028",
    "Poison":   "#A040A0",
    "Ground":   "#E0C068",
    "Flying":   "#A890F0",
    "Psychic":  "#F85888",
    "Bug":      "#A8B820",
    "Rock":     "#B8A038",
    "Ghost":    "#705898",
    "Dragon":   "#7038F8",
    "Dark":     "#705848",
    "Steel":    "#B8B8D0",
    "Fairy":    "#EE99AC",
}
