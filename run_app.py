import streamlit.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=localhost"]
    sys.exit(stcli.main()) 