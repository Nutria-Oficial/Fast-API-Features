# Fast API Features

Features / bibliotecas que serão utilizadas na FastAPI.  
Repositório que reúne exemplos, configurações e libs auxiliares para projetos FastAPI.

---

## 📁 Estrutura do Repositório

```text
.
├── docker/
├── docs/
├── libs/
├── notebooks/
├── .env
├── docker-compose.yml
├── main.py
├── parar-servidor.bat
├── requirements.txt
├── subir-servidor.bat
├── LICENSE
└── README.md
`
```


* *docker/* — arquivos relacionados à containerização (Docker).
* *docs/* — documentação adicional, especificações, guias ou anotações para o projeto.
* *libs/* — bibliotecas ou módulos auxiliares que complementam a FastAPI.
* *notebooks/* — notebooks (por exemplo, Jupyter) para testes, análises ou demonstrações.
* *.env* — variáveis de ambiente para configurar o projeto.
* *docker-compose.yml* — definição de serviços para orquestração com Docker.
* *main.py* — ponto de entrada da aplicação FastAPI.
* *parar-servidor.bat* / *subir-servidor.bat* — scripts em batch para iniciar ou parar o servidor (no Windows).
* *requirements.txt* — dependências Python necessárias.
* *LICENSE* — licença do projeto (MIT).

---

## ⚙️ Pré-requisitos

* Python (3.8+ recomendado)
* Docker & Docker Compose (opcional, para rodar via container)
* Arquivo .env configurado

---

## 🚀 Como rodar localmente

1. Clone o repositório:

   ```bash
   git clone https://github.com/Nutria-Oficial/Fast-API-Features.git
   cd Fast-API-Features
   ```
   

2. Configure o ambiente:

   ```bash
   cp .env.example .env   # se houver exemplo
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Rodar localmente:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   uvicorn main:api --port 8000
   ```

5. (Opcional) Usando Docker:

   ```bash
   docker-compose up --build
   ```

6. (Windows) Scripts para iniciar/parar servidor:

   ```powershell
   .\subir-servidor.bat
   .\parar-servidor.bat
   ```

---

## 🧰 Features / Libs

* Autenticação e segurança
* Integração com banco de dados
* Documentação automática (Swagger / OpenAPI)
* Criação de tabelas nutricionais automática
* Conversas com chat-bot
* Avaliação nutricional automática das tabelas criadas
  
(A lista será expandida conforme novas features forem adicionadas.)

---

## 🧪 Exemplo rápido

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong"}
```

---

## 📦 Licença

Este projeto está sob a licença *MIT*.
Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📝 Commits

Use o padrão abaixo para mensagens de commit:


tipo(modulo): descrição da mudança


### Tipos comuns

| Tipo       | Significado                              |
| ---------- | ---------------------------------------- |
| feat✨    | Nova funcionalidade                      |
| fix🧰     | Correção de bug                          |
| docs🗂️    | Alterações de documentação e estrutura de pastas |
| delete🚯  | Remoção de arquivos ou recursos          |
| perf⚡    | Melhoria de performance sem mudar lógica |
| revert🔄  | Reversão de commit                       |
| merge⤴️ | Merge de branch                          |

---

## 🙋 Contribuindo

1. Abra um issue descrevendo sua ideia
2. Faça um fork do repositório
3. Crie uma branch (git checkout -b minha-feature)
4. Faça commit seguindo o padrão acima
5. Abra um pull request 🚀

---