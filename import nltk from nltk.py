import nltk
from nltk.chat.util import Chat, reflections

# Define predefined patterns and responses
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you', ['I am good, thank you!', 'Doing well, thanks.']),
    (r'what is your name', ['I am a chatbot.', 'You can call me ChatGPT.']),
    (r'quit|bye', ['Goodbye!', 'Bye!', 'See you later.']),
    # Add more patterns and responses as needed
]

# Create a chatbot using the predefined patterns
chatbot = Chat(patterns, reflections)

# Function to start the conversation
def start_chat():
    print("Hello! I'm your chatbot. You can start chatting. Type 'quit' to exit.")
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot.respond(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    # Download NLTK data if not already downloaded
    nltk.download('punkt')

    # Start the chat
    start_chat()

