@echo off
echo ================================================
echo   LangChain Agentic Vision RAG — Auto Setup
echo ================================================
echo.
echo [1/3] Installing Python packages...
python -m pip install -r requirements.txt
echo.
echo [2/3] Checking Ollama (optional)...
where ollama >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Ollama found! Pulling models...
    ollama pull llava
    ollama pull nomic-embed-text
    ollama pull mistral
) ELSE (
    echo Ollama not found. Using Mistral AI API instead.
    echo Get free key at: https://console.mistral.ai
)
echo.
echo [3/3] Setup complete!
echo.
echo Add your Mistral API key to api.env then run:
echo   streamlit run app.py
echo.
pause
