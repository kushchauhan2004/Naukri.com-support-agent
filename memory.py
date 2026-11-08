# Part 2 : LangGraph Agent with Tools, Memory & Guardrails
# Task 8 : Add persisted memory.
import json
import os


# json file for storing conversation history

memory_file = "conversation_memory.json"


# load saved conversations

def load_memory():

    if os.path.exists(memory_file):

        with open(memory_file, "r") as file:
            return json.load(file)

    return {}


# save conversations

def save_memory(memory):

    with open(memory_file, "w") as file:
        json.dump(memory, file, indent=4)


# clear one conversation

def clear_conversation(conversation_id):

    memory = load_memory()

    memory.pop(conversation_id, None)

    save_memory(memory)


# chat with persisted memory

def memory_chat(conversation_id, user_message):

    memory = load_memory()

    if conversation_id not in memory:
        memory[conversation_id] = []

    history = memory[conversation_id]

    message = user_message.lower()

    # check whether the user is asking about previous information

    if "preferred role" in message or "what role did i say" in message:

        previous_message = None

        for item in reversed(history):

            if "preferred role is" in item["user"].lower():

                previous_message = item["user"]
                break

        if previous_message:

            response = (
                "Your preferred role is "
                + previous_message.split("is", 1)[1].strip()
            )

        else:

            response = "I don't have your preferred role in this conversation."

    else:

        response = "I will keep this information in our conversation history."

    # save current turn

    history.append({
        "user": user_message,
        "assistant": response
    })

    memory[conversation_id] = history

    save_memory(memory)

    return response


# demonstrate first conversation

conversation_id = "conversation_1"

clear_conversation(conversation_id)

print("first conversation")

query1 = "My preferred role is Data Analyst."

response1 = memory_chat(conversation_id, query1)

print("user:", query1)
print("assistant:", response1)


# second turn in the same conversation

query2 = "What is my preferred role?"

response2 = memory_chat(conversation_id, query2)

print("\nuser:", query2)
print("assistant:", response2)


# demonstrate a fresh conversation

fresh_conversation_id = "conversation_2"

clear_conversation(fresh_conversation_id)

print("\nfresh conversation")

query3 = "What is my preferred role?"

response3 = memory_chat(fresh_conversation_id, query3)

print("user:", query3)
print("assistant:", response3)


# display saved memory

print("\nsaved conversation history")

with open(memory_file, "r") as file:
    print(json.dumps(json.load(file), indent=4))


# Task 9 : Add a structured output schema.
import json
from jsonschema import validate, ValidationError


# define response schema

response_schema = {
    "type": "object",
    "properties": {
        "conversation_id": {
            "type": "string"
        },
        "user_message": {
            "type": "string"
        },
        "response": {
            "type": "string"
        },
        "memory_used": {
            "type": "boolean"
        },
        "status": {
            "type": "string",
            "enum": ["success", "no_memory", "error"]
        }
    },
    "required": [
        "conversation_id",
        "user_message",
        "response",
        "memory_used",
        "status"
    ],
    "additionalProperties": False
}


# validate agent response

def validate_response(agent_response):

    try:
        validate(
            instance=agent_response,
            schema=response_schema
        )

        print("response validation successful")
        return True

    except ValidationError as error:

        print("response validation failed:", error.message)
        return False


# create structured agent response

def structured_response(
    conversation_id,
    user_message,
    response,
    memory_used,
    status
):

    result = {
        "conversation_id": conversation_id,
        "user_message": user_message,
        "response": response,
        "memory_used": memory_used,
        "status": status
    }

    if validate_response(result):
        return result

    raise ValueError("invalid agent response")


# demonstrate structured output

agent_result = structured_response(
    conversation_id="conversation_1",
    user_message="What is my preferred role?",
    response="Your preferred role is Data Analyst.",
    memory_used=True,
    status="success"
)

print("\nstructured agent response")

print(json.dumps(agent_result, indent=4))


# test invalid response

invalid_result = {
    "conversation_id": "conversation_1",
    "user_message": "What is my preferred role?",
    "response": "Your preferred role is Data Analyst.",
    "memory_used": "yes",
    "status": "success"
}

print("\ninvalid response test")

validate_response(invalid_result)