import random
import torch
from transformers import pipeline

def generate_template_based_story(keywords, genre, min_words=50):
    templates = {
        "fantasy": [
            "Once upon a time in a mystical land, {0} embarked on an adventure filled with {1} and {2}. However, the journey was far from easy. {0} faced numerous challenges, including a powerful sorcerer and an ancient curse. The stars guided {0} through enchanted forests, where whispers of destiny echoed through the trees. In the end, {0} discovered the true meaning of courage and magic.",
            "Legends spoke of {0}, a hero destined to retrieve {1} from the depths of {2}. Many had tried before, but none had returned. Guided by an ancient prophecy, {0} ventured into the heart of darkness, where shadows danced and secrets lurked. With every step, {0} grew stronger, until the final confrontation with fate itself."
        ],
        "horror": [
            "In the dead of night, {0} stumbled upon a {1}, unaware of the {2} lurking in the shadows. The air was thick with fear as whispers called {0}'s name. With each step forward, the darkness seemed to close in, wrapping around {0} like a suffocating embrace. The cursed {1} loomed in the distance, but {0} had no choice but to enter, knowing {2} awaited inside. What lay beyond was beyond nightmares, beyond reason, and beyond escape."
        ],
        "romantic": [
            "Under the shimmering moonlight, {0} met {1}, and their love story unfolded with {2}. Each moment spent together was like a dream, yet the world around them seemed determined to pull them apart. {0} held onto hope, believing that love could defy all odds. With every letter, every whisper, and every stolen glance, the bond between {0} and {1} grew stronger. Would fate be kind, or would love be lost to the tides of time?"
        ]
    }
    
    template_list = templates.get(genre, ["{0} encountered {1} and discovered {2}."])
    template = random.choice(template_list)
    
    while len(keywords) < 3:
        keywords.append(random.choice(["magic", "mystery", "shadow"]))
    
    story = template.format(*keywords[:3])
    
    while len(story.split()) < min_words:
        story += " " + random.choice(["The journey continued.", "Darkness loomed ahead.", "Hope flickered in the distance."])
    
    return story

def generate_gpt_based_text(keywords, genre, min_words=50, max_length=150):
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Write a {genre} story with at least {min_words} words about {', '.join(keywords)}."
    
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    return output[0]['generated_text']

def generate_text(keywords, genre="fantasy", method="gpt", min_words=50):
    if method == "template":
        return generate_template_based_story(keywords, genre, min_words)
    elif method == "gpt":
        return generate_gpt_based_text(keywords, genre, min_words)
    else:
        return "Invalid method selected. Choose 'template' or 'gpt'."

def main():
    print("Welcome to the Story/Poem Generator!")
    min_words = int(input("Enter minimum number of words: ").strip())
    keywords = input("Enter keywords (comma-separated): ").split(',')
    genre = input("Enter genre (fantasy, horror, romantic): ").strip().lower()
    method = input("Choose method (template/gpt): ").strip().lower()
    
    generated_text = generate_text([k.strip() for k in keywords], genre, method, min_words)
    print("\nGenerated Story/Poem:")
    print(generated_text)

if __name__ == "__main__":
    main()