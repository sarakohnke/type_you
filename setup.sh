mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
python -m wget -o model_download.pkl https://type-you.s3.eu-west-2.amazonaws.com/model_pkl.pickle
