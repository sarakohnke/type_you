mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
python -m wget https://type-you.s3.eu-west-2.amazonaws.com/model2.pbz2
