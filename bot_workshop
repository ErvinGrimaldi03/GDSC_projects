import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, ne_chunk, pos_tag
from translate import Translator
import json

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class ChatBot:

    def __init__(self):
        self.previous_sentiment = None
        self.user_authenticated = False
        self.user_name = None
        self.schedule = {}
        self.sia = SentimentIntensityAnalyzer()
        self.translator = Translator(to_lang="en")  # Default translation to English

    def analyze_sentiment(self, text_content):
        sentiment_score = self.sia.polarity_scores(text_content)['compound']
        return sentiment_score

    def translate_text(self, text, target_language):
        translator = Translator(to_lang=target_language)
        translation = translator.translate(text)
        return text, translation

    def authenticate_user(self, user_input):
        tree = ne_chunk(pos_tag(word_tokenize(user_input)))
        for subtree in tree.subtrees():
            if subtree.label() == 'PERSON':
                self.user_name = ' '.join([t[0] for t in subtree.leaves()])
                self.user_authenticated = True
                return f"Welcome, {self.user_name}!"

        return None

    def manage_schedule(self, user_input):
        words = user_input.split()
        if "schedule" in words:
            try:
                event_idx = words.index("schedule") + 1
                time_idx = words.index("at") + 1
                event = words[event_idx:time_idx-1]
                time = words[time_idx:]
                self.schedule[" ".join(event)] = " ".join(time)
                return f"{self.user_name}, '{' '.join(event)}' has been scheduled at {' '.join(time)}."
            except ValueError:
                return "I couldn't understand the schedule. Please use the format: schedule [event] at [time]."
        return None

    def get_schedule(self):
        if not self.schedule:
            return f"{self.user_name}, you don't have anything scheduled."
        else:
            schedule_str = ', '.join([f"'{event}' at {time}" for event, time in self.schedule.items()])
            return f"{self.user_name}, here's your schedule: {schedule_str}"

    def save_state(self, filename="bot_state.json"):
        data = {
            'user_name': self.user_name,
            'schedule': self.schedule
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print("State saved successfully!")

    def load_state(self, filename="bot_state.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.user_name = data.get('user_name', None)
                self.schedule = data.get('schedule', {})
                self.user_authenticated = True if self.user_name else False
            print("State loaded successfully!")
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with a fresh state.")

    def retrive_schedules(self):
        return self.get_schedule()

    def generate_response(self, user_input, target_language="en"):
        # Translate to English for processing
        _, translated_input = self.translate_text(user_input, "en")

        # User authentication
        if not self.user_authenticated:
            auth_response = self.authenticate_user(translated_input)
            if auth_response:
                _, translated_response = self.translate_text(auth_response, target_language)
                return translated_response
            else:
                _, translated_response = self.translate_text("Please authenticate by telling me your name.", target_language)
                return translated_response

        # Schedule management
        schedule_response = self.manage_schedule(translated_input)
        if schedule_response:
            _, translated_response = self.translate_text(schedule_response, target_language)
            return translated_response

        # Rule-based part
        if "hello" in translated_input.lower():
            response = f"Hi {self.user_name}! How can I help you today?"
        elif "bye" in translated_input.lower():
            response = "Goodbye! Have a great day!"
        else:
            sentiment_score = self.analyze_sentiment(translated_input)

            response = ""
            if sentiment_score > 0:
                response += "You seem happy! "
            elif sentiment_score < 0:
                response += "You seem upset. "
            else:
                response += "Your message seems neutral. "

            if self.previous_sentiment is not None:
                if self.previous_sentiment > 0 and sentiment_score <= 0:
                    response += "You were happy before, what happened? "
                elif self.previous_sentiment <= 0 and sentiment_score > 0:
                    response += "Glad to see your mood uplifted! "

            response += "How can I assist you further?"

        # Check for user identity retrieval command
        if "who am i" in translated_input.lower():
            if self.user_authenticated:
                return f"You are {self.user_name}."
            else:
                return "I don't know who you are. Please authenticate by telling me your name."

        # Handle state saving command
        if "save my data" in translated_input.lower():
            self.save_state()
            return "Your data has been saved."

        # Handle state loading command
        if "load my data" in translated_input.lower():
            self.load_state()
            return "Your data has been loaded."

        if "show my schedule" in translated_input.lower():
            return self.get_schedule()

        if "show all schedules" in translated_input.lower():
            return self.retrive_schedules()

        # Translate response back to the original language
        _, translated_response = self.translate_text(response, target_language)
        return translated_response

# set schedule : schedule [event] at [time]
# retrive schedule : show all schedules


# Example usage:
bot = ChatBot()
#bot.load_state()  # Load the previous state when the bot starts
while True:
    user_input = input("You: ")
    print("Bot:", bot.generate_response(user_input, target_language="en") )  # English responses.
    #print("Bot:", bot.generate_response(user_input, target_language="es"))  # Spanish responses.
