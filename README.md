# vakalat-ai

# Fetching data from IndianKanoon : (https://api.indiankanoon.org/documentation/)

## Search for documents about "contract law"
python a.py -s YOUR_TOKEN -D ./data -q "contract law"

## Download a specific document
python a.py -s YOUR_TOKEN -D ./data -d 12345

## Download document fragment matching a query
python a.py -s YOUR_TOKEN -D ./data -d 12345 -q "breach of contract"

## Download all Supreme Court judgments from 2023
python a.py -s YOUR_TOKEN -D ./data -c "supremecourt" -f "01-01-2023" -t "31-12-2023"
