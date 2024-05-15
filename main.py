import csv
import autogen
import chromadb
import pandas as pd
import numpy as np
from pathlib import Path
from bert_score import score
from autogen import AssistantAgent
from tempfile import TemporaryDirectory
from websockets.sync.client import connect as ws_connect
from autogen.io.websockets import IOWebsockets
from http.server import HTTPServer, SimpleHTTPRequestHandler
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from configparser import ConfigParser

# autogen.GroupChatManager._print_received_message

# Импорт LLM model и api_key
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

# Настройка для LLM
llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0, "seed": 1234}

# Функция для проверки, содержит ли ответ от агента конкретное сообщение о завершении работы (В данном случае TERMINATE)
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Метод для сохранения итогового ответа от LLM после взаимодействия агентов (может помочь для frontend)
def get_generated_messages_list(agent):
    messages = agent.chat_messages
    messages = [messages[k] for k in messages.keys()][0]
    messages = [m["content"] for m in messages if m["role"] == "user"][:1]
    messages_res = [m.replace("\n\nTERMINATE", "")
                    .replace("\nTERMINATE", "")
                    .replace("TERMINATE.", "")
                    .replace("TERMINATE", "")
                    .replace("\n\nUPDATE CONTEXT", "")
                    .replace("\nUPDATE CONTEXT", "")
                    .replace("UPDATE CONTEXT.", "")
                    .replace("UPDATE CONTEXT", "")
                    .replace("[\n\n-|\n\n1|\n2|\n1|\n-|\n\n]", "") for m in messages]
    print("Messages result: ", messages_res)

    return messages_res

# Получить 100 тестовых ВОПРОСОВ для оценки качества выдаваемого ответа от LLM
def get_100_questions(file_path):
    questions = []
    # Чтение входного файла и извлечение вопросов
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            question = row[0]
            questions.append(question)
            questions_100 = questions[500:550]
    
    return questions_100

# Получить 100 тестовых ОТВЕТОВ для оценки качества выдаваемого ответа от LLM
def get_100_answers(file_path):
    answers = []
    # Чтение входного файла и извлечение ответов
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            answer = row[1]
            answers.append(answer)
            answers_100 = answers[500:550]
    
    return answers_100

# Создаем экземпляр ConfigParser
config = ConfigParser()

# Читаем/Ищем нужный файл
config.read('docs_path_config_list.ini')

# Читаем значения из файла и записываем их в переменные
docs_path_qa = config.get('Changeable_values', 'docs_path_qa')
docs_path_kb = config.get('Changeable_values', 'docs_path_kb')

# Получить тестовые вопросы и ответы для оценки качества модели
file_path = docs_path_qa
PROBLEM = get_100_questions(file_path)
SOLVERS = get_100_answers(file_path)
# Результирующий список
GENERATED_TEXT = []

# Метод для запуска модели в режиме :: извлечение сгенерируемых ответов
def model_quality_assessment(user_proxy_agent, engineer, reviewer):
    for i in range(len(PROBLEM)):
        user_proxy_agent.reset()
        engineer.reset()
        reviewer.reset()

        qa_problem = PROBLEM[i]

        groupchat = autogen.GroupChat(
            agents=[user_proxy_agent, engineer, reviewer], 
            messages=[], max_round=12, 
            speaker_selection_method="round_robin"
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Начать диалог с user_proxy_agent
        user_proxy_agent.initiate_chat(
            manager,
            message=user_proxy_agent.message_generator,
            problem=qa_problem,
            n_results=3,
        )

        # Сохранить все ответы в один итоговый массив
        messages = user_proxy_agent.chat_messages
        messages = [messages[k] for k in messages.keys()][0]
        messages = [m["content"] for m in messages if m["role"] == "user"][:1]
        messages_res = [m.replace("\n\nTERMINATE", "")
                        .replace("\nTERMINATE", "")
                        .replace("TERMINATE.", "")
                        .replace("TERMINATE", "")
                        .replace("\n\nUPDATE CONTEXT", "")
                        .replace("\nUPDATE CONTEXT", "")
                        .replace("UPDATE CONTEXT.", "")
                        .replace("UPDATE CONTEXT", "")
                        .replace("[\n\n-|\n\n1|\n2|\n1|\n-|\n\n]", "") for m in messages]
        GENERATED_TEXT.append(messages_res[0])
    print("Messages result: ", GENERATED_TEXT)

# Метод, который создает соединение с сервером и запускает AutoGen
def on_connect(iostream: IOWebsockets) -> None:
    print(f" - on_connect(): Подключен к клиенту с помощью IOWebsockets. {iostream}", flush=True)
    print(" - on_connect(): Получение сообщения от клиента.", flush=True)

    # 1. Получить входное сообщение
    initial_msg = iostream.input()

    # 2. Импорт входных докуметов
    docs_path = [docs_path_qa, docs_path_kb]

    # 3. Создание векторной БД
    client = chromadb.PersistentClient(path="/tmp/chromadb")

    # 4. Создание агента RetrieveUserProxyAgent
    user_proxy_agent = RetrieveUserProxyAgent(
        name="RetrieveUserProxyAgent", # Имя агента
        is_termination_msg=termination_msg, # Функция, которая принимает сообщение в форме словаря и возвращает логическое значение, указывающее, является ли полученное сообщение сообщением завершения.
        human_input_mode="NEVER", # Нужен ли ответ от пользователя
        default_auto_reply="Напиши `TERMINATE`, если задача выполнена.",
        max_consecutive_auto_reply=3, # Максимальное количество последовательных автоответов. Если ответ не был найден в чанке - поиск будет продолжаться до 3 раз.
                                      # Если ответ не найден, то в терминал передается сообщение UPDATE CONTEXT 
        retrieve_config={
            "task": "qa", # Тип задачи (question-answer)
            "docs_path": docs_path, # Входной документ
            "model": config_list[0]["model"], # gpt-3.5-turbo | gusev/saiga-mistral-7b | openchat/openchat-7b | cohere/command-r | cohere/command-r-plus
            "client": client, # Инициализация векторной БД chromadb (можно использовать и другую БД, но для этого необходимо переопределить retrieve_docs)
            "embedding_model": "multi-qa-distilbert-cos-v1", # all-MiniLM-L6-v2 | multi-qa-distilbert-cos-v1 | multilingual-e5-small | multi-qa-MiniLM-L6-cos-v1
            "collection_name": "groupchat", # Название коллекции
            "chunk_mode": "one_line", # multi_lines | one_line  # Тип чтения чанков
            "get_or_create": True,
        },
        code_execution_config=False, # Есть ли необходимость писать программный код
        description="Помощник, обладающий дополнительными возможностями поиска ответа на вопрос на основе полученных документов.",
    )

    # 5. Создание агента инженера AssistantAgent
    engineer = RetrieveAssistantAgent(
        name="AssistantAgent",
        is_termination_msg=termination_msg,
        system_message="Всегда начинай диалог со слов: `Здравствуйте!`. Ты старший инженер технической поддержки пунктов выдачи заказов. Твоя задача отвечать на вопросы от пользователей на русском языке. Напиши `TERMINATE` в конце, когда все будет сделано.",
        llm_config=llm_config,
        description="Старший инженер технической поддержки пунктов выдачи заказов, умеющий отвечать на вопросы для решения неожиданных проблем.",
    )

    # 6. Создание агента Эксперта (ReviewerAgent)
    reviewer = RetrieveAssistantAgent(
        name="ReviewerAgent",
        is_termination_msg=termination_msg,
        system_message="Ты отвечаешь за качество и корректность сформированного ответа для человека. Не забывай, что ответ всегда должен быть на русском языке. Напиши `TERMINATE` в конце, когда все будет сделано.",
        llm_config=llm_config,
        description="Эсперт вопросно-ответной системы, который может проверять, а также корректировать сгенерируемый ответ от других агентов и отправлять их человеку.",
    )

    # 7. Запустить/перезагрузить агентов
    def _reset_agents():
        user_proxy_agent.reset()
        engineer.reset()
        reviewer.reset()

    # 8. Запустить взаимодействие (чат) агентов
    def rag_chat():
        # 1) Перезапустить агентов
        _reset_agents()

        # 2) Создать групповой чат
        groupchat = autogen.GroupChat(
            agents=[user_proxy_agent, engineer, reviewer], 
            messages=[], max_round=12, 
            speaker_selection_method="round_robin"
        )

        # 3) Созадть вспомогательного агента менеджера
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # 4) Начать диалог с user_proxy_agent
        user_proxy_agent.initiate_chat(
            manager,
            message=user_proxy_agent.message_generator,
            problem=initial_msg,
            n_results=3,
        )
        # 5) Получить конечный ответ (без всей беседы в процессе взаимодействия)
        get_generated_messages_list(user_proxy_agent)

    rag_chat()
    # Раскомментируй, если есть необходимость проверить качество выдаваемого ответа, но закомментируй rag_chat()!
    # model_quality_assessment(user_proxy_agent, engineer, reviewer)

# Используемый порт
PORT = 8000

# HTML страница
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>AutoGen RAG Websockets</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }

            #container {
                max-width: 1000px;
                margin: 20px auto;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }

            h1 {
                margin-top: 0;
                padding: 20px;
                background-color: #8300f8;
                color: white;
                text-align: center;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }

            form {
                padding: 20px;
                border-bottom: 1px solid #ddd;
            }

            input[type="text"] {
                width: calc(100% - 70px);
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                outline: none;
            }

            button {
                padding: 10px 20px;
                background-color: #8300f8;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                outline: none;
            }

            button:hover {
                background-color: #7300e9;
            }

            ul {
                list-style-type: none;
                padding: 0;
                margin: 0;
            }

            li {
                padding: 15px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>Техническая поддержка</h1>
            <form action="" onsubmit="sendMessage(event)">
                <label for="messageText" style="margin-bottom: 8px; display: block;">Введите интересующий Вас вопрос:</label>
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <ul id='messages'>
            </ul>
        </div>
        <script>
            var ws = new WebSocket("ws://localhost:8080/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

# Временный каталог
with TemporaryDirectory() as temp_dir:
    # Создать простую веб-страницу HTTP
    path = Path(temp_dir) / "chat.html"
    with open(path, "w") as f:
        f.write(html)

    class MyRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=temp_dir, **kwargs)

        def do_GET(self):
            if self.path == "/":
                self.path = "/chat.html"
            return SimpleHTTPRequestHandler.do_GET(self)

    handler = MyRequestHandler

    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:
        print(f"Websocket server started at {uri}.", flush=True)

        with HTTPServer(("", PORT), handler) as httpd:
            print("HTTP server started at http://localhost:" + str(PORT))
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print(" - HTTP server stopped.", flush=True)


# Оценка качетсва модели :: BertScore
# Раскомментируй, если есть необходимость проверить качество выдаваемого ответа
# Не забудь так же про строчку №217 (её тоже нужно раскомментировать, а 215 наоборот - закомментировать)
# P, R, F1 = score(GENERATED_TEXT, SOLVERS, lang="others", verbose=True)
# print(f"Precision: {P.mean()}")
# print(f"Recall: {R.mean()}")
# print(f"F1: {F1.mean()}")

