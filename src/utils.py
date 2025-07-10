# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
import sys
import warnings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


warnings.filterwarnings("ignore", category=UserWarning, module='langchain_experimental.agents.agent_toolkits.pandas.base')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_and_preprocess_data(file_path):
    """Loads and preprocesses a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = df.columns.str.replace(' ', '_').str.lower()

        for col in df.columns:
            if 'date' in col or 'time' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                try:
                    if df[col].dtype == 'object':
                        temp_col = pd.to_numeric(df[col], errors='coerce')
                        if (temp_col.count() / df[col].count()) > 0.5:
                            df[col] = temp_col
                except (ValueError, TypeError):
                    pass

        df.dropna(how='all', inplace=True)
        print("Data loaded and preprocessed. Here's a look at the first 5 rows:")
        print(df.head())
        print("\nDataFrame Info (important for understanding columns and data types):")
        df.info()
        print(f"\nDataFrame shape (rows, columns): {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}. Please ensure it's a valid CSV file.")
        return None

def find_gemini_model():
    """Finds a suitable Gemini model for text generation."""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if m.name.startswith('models/gemini') and '-vision' not in m.name:
                    available_models.append(m.name)
        if not available_models:
            print("No suitable Gemini text models found supporting 'generateContent'.")
            return None
        for preferred_model in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-2.0-flash', 'models/gemini-1.0-pro']:
            if preferred_model in available_models:
                return preferred_model.replace('models/', '')
        return available_models[0].replace('models/', '')
    except Exception as e:
        print(f"Error listing models: {e}.")
        return None

def setup_llm_and_agent(dataframe):
    """Sets up the LLM and Pandas DataFrame Agent."""
    if dataframe is None:
        print("Cannot create agent: DataFrame is None.")
        return None, "N/A"

    model_name = find_gemini_model()
    if model_name is None:
        print("Could not find a suitable Gemini model. Agent not initialized.")
        return None, "N/A"

    print(f"\nInitializing Gemini LLM ({model_name})...")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)

    df_description = f"""
    You are an expert data analyst. You are interacting with a pandas DataFrame in Python.
    The DataFrame is named `df`.

    Here's information about the DataFrame:
    - Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns.
    - Columns and their dtypes:
    """
    for col, dtype in dataframe.dtypes.items():
        df_description += f"  - `{col}`: {dtype}\n"

    df_description += "\nHere are the first 5 rows of the DataFrame:\n"
    df_description += dataframe.head().to_string()
    df_description += "\n\nBased on this information, answer my questions using Pandas operations and Python code. "
    df_description += "If I ask for a visualization, generate Python code using `matplotlib.pyplot` (imported as `plt`) and `seaborn` (imported as `sns`). "
    df_description += "Ensure you use the DataFrame `df` directly in your code. "
    df_description += "Always include `import matplotlib.pyplot as plt` and `import seaborn as sns` at the beginning of your plotting code. "
    df_description += "When generating plotting code, **do NOT include `plt.show()`** as it will be handled automatically. "
    df_description += "If you generate a plot, provide only the code in your response without any additional text."
    df_description += "If you are answering a textual question, provide the answer in natural language."

    print("\nCreating Pandas DataFrame Agent...")
    llm_agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        agent_kwargs={
            "prefix": df_description
        }
    )
    print("Pandas DataFrame Agent created successfully.")
    return llm_agent, model_name

@contextlib.contextmanager
def capture_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_catcher = io.StringIO()
    sys.stderr = stderr_catcher = io.StringIO()
    try:
        yield stdout_catcher, stderr_catcher
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def get_insights(query: str, llm_agent, df_global):
    print(f"\nUser Query: {query}")
    print("AI (Thinking...):")

    if llm_agent is None:
        print("Error: LLM Agent is not initialized.")
        return "Error: Agent not ready."

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            response_obj = llm_agent.invoke({"input": query})
            raw_response = response_obj['output']

        agent_stdout = captured_stdout.getvalue()
        agent_stderr = captured_stderr.getvalue()

        if agent_stdout:
            print("\n--- Agent's Internal Process ---")
            print(agent_stdout)
            print("--------------------------------")
        if agent_stderr:
            print("\n--- Agent's Internal Errors/Warnings ---")
            print(agent_stderr)
            print("---------------------------------------")

        if "```python" in raw_response:
            try:
                code_block = raw_response.split("```python")[1].split("```")[0].strip()
                code_block = code_block.replace("plt.show()", "")

                print("\nAI generated plotting code. Executing now...\n")

                exec_context = {'df': df_global, 'plt': plt, 'sns': sns}
                plt.clf()

                with capture_output() as (stdout, stderr):
                    exec(code_block, globals(), exec_context)

                plot_stdout = stdout.getvalue()
                plot_stderr = stderr.getvalue()

                if plot_stdout:
                    print("\n--- Plot Code Output (Stdout) ---")
                    print(plot_stdout)
                    print("---------------------------------")
                if plot_stderr:
                    print("\n--- Plot Code Errors/Warnings (Stderr) ---")
                    print(plot_stderr)
                    print("------------------------------------------")

                if plt.gcf()._axstack is not None:
                    print("\nPlot generated successfully! (Displayed above)")
                else:
                    print("AI generated plotting code, but no plot was actually created. Please check the code or rephrase.")
                    print(f"Raw Code:\n```python\n{code_block}\n```")
                    print(f"Stdout from plot execution:\n{plot_stdout}")
                    print(f"Stderr from plot execution:\n{plot_stderr}")
                    return "AI generated plotting code, but no plot was actually created. Check the console for generated code and errors."

                return "Plot generated above! If you don't see it, try adjusting your query."

            except Exception as plot_err:
                print(f"\nError executing generated plot code: {plot_err}")
                print(f"AI's raw response:\n```python\n{raw_response}\n```")
                print(f"Stdout from plot execution:\n{plot_stdout}")
                print(f"Stderr from plot execution:\n{plot_stderr}")
                return f"An error occurred while trying to generate the plot: {plot_err}. AI's raw code was provided in the console. Please try rephrasing."
        else:
            print(f"AI Final Answer: {raw_response}")
            return raw_response

    except Exception as agent_err:
        print(f"An error occurred during agent invocation: {agent_err}")
        return f"An error occurred: {agent_err}. Please try rephrasing your question or check the console for details."