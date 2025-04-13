# The Words for LLM's

![Zrzut ekranu 2025-04-13 144552](https://github.com/user-attachments/assets/d6c45804-1c75-485f-988d-358a4db681ca)

A proxy service that integrates LLMs (Large Language Models) with webhook automation, featuring keyword-based triggers and a web-based management interface.

*Usługa proxy integrująca modele LLM (Large Language Models) z automatyzacją webhooków, z funkcją wyzwalaczy opartych na słowach kluczowych i webowym interfejsem zarządzania.*

## 🌍 Languages / Języki

- [English](#english)
- [Polski](#polski)

---

<a name="english"></a>
## English

### 📋 Overview

The Words for LLM's is a proxy service that sits between your applications and LLM backends (such as Llama.cpp or Ollama). It adds powerful keyword detection capabilities to automatically trigger webhook actions based on specific words found in user queries. The system includes a web-based dashboard for easy management of keywords, configuration settings, and monitoring of service status.

### ✨ Features

- **LLM Proxy Service**: Compatible with Llama.cpp and Ollama backends
- **Keyword Trigger System**: Define keywords that trigger specific webhook actions
- **Webhook Integration**: Works with n8n or Flowise for automation workflows
- **OpenAI API Compatibility**: Compatible with applications that use OpenAI's API format
- **Web Dashboard**: Easy management of keywords and configuration
- **Bilingual Interface**: Available in English and Polish

### 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, Alpine.js, TailwindCSS
- **Web Server**: Flask for serving the web interface
- **Database**: SQLite
- **Containerization**: Docker and Docker Compose

### 📊 Screenshots
![Zrzut ekranu 2025-04-13 144552](https://github.com/user-attachments/assets/f47afab0-235a-45ff-a10b-dd39d20f7501)
*Dashboard view showing system status*

![Zrzut ekranu 2025-04-13 144621](https://github.com/user-attachments/assets/16b67854-7fe5-4f06-8593-fded7e27d37b)
*Keywords management interface*

![Zrzut ekranu 2025-04-13 144724](https://github.com/user-attachments/assets/5f213b7b-18de-483e-83f8-df1e218ffe2b)
![Zrzut ekranu 2025-04-13 144740](https://github.com/user-attachments/assets/2de39809-103d-4bae-a4fe-72a9fafeb06a)
![Zrzut ekranu 2025-04-13 144802](https://github.com/user-attachments/assets/aecda3c9-77db-49a9-a898-7f617111203a)
![Zrzut ekranu 2025-04-13 144823](https://github.com/user-attachments/assets/f8dfb0e1-b8c4-43f1-b7c6-6855800db516)

*System configuration settings*

### 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PPTG/The-Words-for-LLM-s.git
   cd The-Words-for-LLM-s
   ```

2. Configure the environment variables in `docker-compose.yml` for your setup.

3. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at `http://localhost:5555`

### 📝 Configuration

The system can be configured via the web interface or by directly editing the database. Main configuration options include:

- **LLM Backend**: Choose between Llama.cpp or Ollama
- **Webhook Service**: Configure n8n or Flowise for automation
- **Model Settings**: Set temperature, max tokens, and other LLM parameters
- **Advanced Settings**: Configure timeouts and debug modes

### 🔍 How It Works

1. User queries are sent to the proxy service
2. The system scans the messages for defined keywords
3. If a keyword is detected, the message is routed to the configured webhook
4. If no keywords match, the message is processed by the configured LLM backend
5. The response is returned to the user

### 🔌 API Endpoints

The service provides several API endpoints:
- Standard chat endpoints: `/api/chat`, `/completion`
- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/models`
- Management endpoints: `/api/keywords`, `/api/config`, `/api/health`

### 🎯 Use Cases

- Enhancing LLM responses with external data from APIs
- Creating customized responses for specific user queries
- Building automated workflows triggered by natural language
- Monitoring and analyzing user queries

---

<a name="polski"></a>
## Polski

### 📋 Przegląd

The Words for LLM's to usługa proxy działająca pomiędzy aplikacjami a backendami LLM (takimi jak Llama.cpp lub Ollama). Dodaje zaawansowane możliwości wykrywania słów kluczowych, aby automatycznie wyzwalać akcje webhooków na podstawie konkretnych słów znalezionych w zapytaniach użytkowników. System zawiera webowy panel administracyjny do łatwego zarządzania słowami kluczowymi, ustawieniami konfiguracji i monitorowania stanu usługi.

### ✨ Funkcje

- **Usługa Proxy LLM**: Kompatybilna z backendami Llama.cpp i Ollama
- **System Wyzwalaczy Opartych na Słowach Kluczowych**: Zdefiniuj słowa kluczowe, które uruchamiają określone akcje webhooków
- **Integracja Webhooków**: Współpracuje z n8n lub Flowise dla przepływów automatyzacji
- **Kompatybilność z API OpenAI**: Kompatybilny z aplikacjami używającymi formatu API OpenAI
- **Panel Webowy**: Łatwe zarządzanie słowami kluczowymi i konfiguracją
- **Dwujęzyczny Interfejs**: Dostępny w języku angielskim i polskim

### 🛠️ Technologie

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, Alpine.js, TailwindCSS
- **Serwer WWW**: Flask do obsługi interfejsu webowego
- **Baza danych**: SQLite
- **Konteneryzacja**: Docker i Docker Compose

### 📊 Zrzuty ekranu

![Zrzut ekranu 2025-04-13 144552](https://github.com/user-attachments/assets/f47afab0-235a-45ff-a10b-dd39d20f7501)
*Widok panelu pokazujący status systemu*

![Zrzut ekranu 2025-04-13 144621](https://github.com/user-attachments/assets/16b67854-7fe5-4f06-8593-fded7e27d37b)
*Interfejs zarządzania słowami kluczowymi*

![Zrzut ekranu 2025-04-13 144724](https://github.com/user-attachments/assets/5f213b7b-18de-483e-83f8-df1e218ffe2b)
![Zrzut ekranu 2025-04-13 144740](https://github.com/user-attachments/assets/2de39809-103d-4bae-a4fe-72a9fafeb06a)
![Zrzut ekranu 2025-04-13 144802](https://github.com/user-attachments/assets/aecda3c9-77db-49a9-a898-7f617111203a)
![Zrzut ekranu 2025-04-13 144823](https://github.com/user-attachments/assets/f8dfb0e1-b8c4-43f1-b7c6-6855800db516)
*Ustawienia konfiguracji systemu*

### 🚀 Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/PPTG/The-Words-for-LLM-s.git
   cd The-Words-for-LLM-s
   ```

2. Skonfiguruj zmienne środowiskowe w pliku `docker-compose.yml` dla swojej instalacji.

3. Uruchom usługi za pomocą Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Dostęp do interfejsu webowego pod adresem `http://localhost:5555`

### 📝 Konfiguracja

System można skonfigurować za pomocą interfejsu webowego lub poprzez bezpośrednią edycję bazy danych. Główne opcje konfiguracyjne obejmują:

- **Backend LLM**: Wybór między Llama.cpp a Ollama
- **Usługa Webhook**: Konfiguracja n8n lub Flowise do automatyzacji
- **Ustawienia Modelu**: Ustawienie temperatury, maksymalnej liczby tokenów i innych parametrów LLM
- **Ustawienia Zaawansowane**: Konfiguracja limitów czasu i trybów debugowania

### 🔍 Jak to działa

1. Zapytania użytkowników są wysyłane do usługi proxy
2. System skanuje wiadomości w poszukiwaniu zdefiniowanych słów kluczowych
3. Jeśli wykryto słowo kluczowe, wiadomość jest kierowana do skonfigurowanego webhooka
4. Jeśli nie znaleziono pasujących słów kluczowych, wiadomość jest przetwarzana przez skonfigurowany backend LLM
5. Odpowiedź jest zwracana do użytkownika

### 🔌 Endpointy API

Usługa udostępnia kilka endpointów API:
- Standardowe endpointy czatu: `/api/chat`, `/completion`
- Endpointy kompatybilne z OpenAI: `/v1/chat/completions`, `/v1/models`
- Endpointy zarządzania: `/api/keywords`, `/api/config`, `/api/health`

### 🎯 Przypadki użycia

- Wzbogacanie odpowiedzi LLM o zewnętrzne dane z API
- Tworzenie niestandardowych odpowiedzi na konkretne zapytania użytkowników
- Budowanie zautomatyzowanych przepływów pracy wyzwalanych przez język naturalny
- Monitorowanie i analizowanie zapytań użytkowników

---

## 📄 License / Licencja

MIT License / Licencja MIT
