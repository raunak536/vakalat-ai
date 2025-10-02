# =============================================================================
# INDIAN KANOON API CLIENT - A tool to download legal documents from India
# =============================================================================
# This script helps lawyers, researchers, and students download court judgments,
# legal documents, and case law from the Indian Kanoon database (api.indiankanoon.org)
# 
# What it does:
# - Searches for legal documents using keywords
# - Downloads court judgments and legal papers
# - Organizes downloaded files in folders by court and date
# - Can process multiple searches at the same time for faster downloads
# =============================================================================

# Import necessary Python libraries for different functions:
import argparse      # For handling command-line arguments (like -q for query)
import logging       # For recording what the program is doing (logs)
import os           # For working with files and folders on your computer
import re           # For searching and matching text patterns
import codecs       # For reading/writing files with special characters
import json         # For handling data in JSON format (like API responses)
import http.client  # For making internet connections to download data
import urllib.request, urllib.parse, urllib.error  # For handling web URLs and encoding
import base64       # For decoding downloaded files (some files are encoded)
import glob         # For finding files that match certain patterns
import csv          # For creating spreadsheet-like files to track downloads
import datetime     # For working with dates and times
import time         # For adding delays between requests (to be nice to the server)
import multiprocessing  # For running multiple downloads at the same time

def print_usage(progname):
    """Show how to use this program with basic command line options"""
    print ('''python %s -t token -o offset -n limit -d datadir''' % progname)

# =============================================================================
# IKApi CLASS - The main class that talks to the Indian Kanoon website
# =============================================================================
# This class handles all communication with the Indian Kanoon API.
# Think of it as a "messenger" that sends requests to the website and gets back
# legal documents, search results, and other information.
# =============================================================================
class IKApi:
    def __init__(self, args, storage):
        """
        Initialize the API client with user settings and preferences.
        This is like setting up your account and preferences before using the service.
        """
        # Set up logging so we can see what the program is doing
        self.logger     = logging.getLogger('ikapi')

        # Set up authentication headers - this is like your login credentials
        # The API needs a special token to know you're allowed to download documents
        self.headers    = {'Authorization': 'Token %s' % args.token, \
                           'Accept': 'application/json'}

        # The website address we'll be talking to
        self.basehost   = 'api.indiankanoon.org'
        
        # The file storage system that will save downloaded documents
        self.storage    = storage
        
        # User preferences for how many documents to download:
        self.maxcites   = args.maxcites      # Maximum number of cases this document cites
        self.maxcitedby = args.maxcitedby    # Maximum number of cases that cite this document
        self.orig       = args.orig          # Whether to download original PDF files too
        self.maxpages   = args.maxpages      # Maximum number of search result pages to download
        self.pathbysrc  = args.pathbysrc     # Whether to organize files by court source
        
        # Settings for parallel processing (downloading multiple things at once):
        self.queue      = multiprocessing.Queue(20)  # A queue to hold tasks for workers
        self.num_workers= args.numworkers            # How many download workers to use
        
        # Date and sorting preferences:
        self.addedtoday = args.addedtoday    # Only get documents added today
        self.fromdate   = args.fromdate      # Start date for search
        self.todate     = args.todate        # End date for search
        self.sortby     = args.sortby        # How to sort results (newest first, etc.)

        # Safety limit: don't download more than 100 pages at once
        if self.maxpages > 100:
            self.maxpages = 100

    def call_api_direct(self, url):
        """
        Make a direct request to the Indian Kanoon API.
        This is like making a phone call to ask for a specific document.
        """
        # Connect to the Indian Kanoon website
        connection = http.client.HTTPSConnection(self.basehost)
        
        # Send a request asking for the document at the given URL
        connection.request('POST', url, headers = self.headers)
        
        # Get the response (the document or data we asked for)
        response = connection.getresponse()
        results = response.read()

        # Convert the response to text if it's in binary format
        if isinstance(results, bytes):
            results = results.decode('utf8')
        return results 
   
    def call_api(self, url):
        """
        Make a request to the API with automatic retry if it fails.
        Sometimes the internet is slow or the server is busy, so we try up to 10 times
        with increasing delays between attempts.
        """
        count = 0

        # Try up to 10 times if something goes wrong
        while count < 10:
            try:
                # Try to get the data from the API
                results = self.call_api_direct(url)
            except Exception as e:
                # If there's an error (like network problem), log it and try again
                self.logger.warning('Error in call_api %s %s', url, e)
                count += 1
                time.sleep(count * 100)  # Wait longer each time (100ms, 200ms, 300ms...)
                continue

            # Check if we got an error message from the server
            if results == None or (isinstance(results, str) and \
                                   re.match('error code:', results)):
                self.logger.warning('Error in call_api %s %s', url, results)
                count += 1
                time.sleep(count * 100)  # Wait and try again
            else:
                # Success! We got the data we wanted
                break 

        return results

    def fetch_doc(self, docid):
        """
        Download a complete legal document by its ID number.
        This gets the full text of a court judgment or legal document.
        """
        # Build the URL to request the document
        url = '/doc/%d/' % docid

        # Add optional parameters to limit how much citation data to include
        args = []
        if self.maxcites > 0:
            args.append('maxcites=%d' % self.maxcites)  # Limit cases this document cites

        if self.maxcitedby > 0:
            args.append('maxcitedby=%d' % self.maxcitedby)  # Limit cases that cite this document

        # Add the parameters to the URL if any were specified
        if args:
            url = url + '?' + '&'.join(args)

        return self.call_api(url)

    def fetch_docmeta(self, docid):
        """
        Get metadata (information about) a document without downloading the full text.
        This is like getting a book's title, author, and summary without reading the whole book.
        """
        url = '/docmeta/%d/' % docid

        # Add the same citation limits as for full documents
        args = []
        if self.maxcites != 0:
            args.append('maxcites=%d' % self.maxcites)

        if self.maxcitedby != 0:
            args.append('maxcitedby=%d' % self.maxcitedby)

        if args:
            url = url + '?' + '&'.join(args)

        return self.call_api(url)

    def fetch_orig_doc(self, docid):
        """
        Download the original document file (usually a PDF) from the court.
        This is the actual scanned document as it was filed with the court.
        """
        url = '/origdoc/%d/' % docid
        return self.call_api(url)

    def fetch_doc_fragment(self, docid, q):
        """
        Get only the parts of a document that match a specific search query.
        Instead of downloading the whole document, this finds just the relevant sections.
        """
        # Encode the search query so it can be safely sent in a URL
        q   = urllib.parse.quote_plus(q.encode('utf8'))
        url = '/docfragment/%d/?formInput=%s' % (docid,  q)
        return self.call_api(url)

    def search(self, q, pagenum, maxpages):
        """
        Search for documents matching a query and get a list of results.
        This is like using a search engine to find relevant legal documents.
        """
        # Encode the search query for the URL
        q = urllib.parse.quote_plus(q.encode('utf8'))
        url = '/search/?formInput=%s&pagenum=%d&maxpages=%d' % (q, pagenum, maxpages)
        return self.call_api(url)


    def save_doc_fragment(self, docid, q):
        """
        Download and save only the parts of a document that match a search query.
        This is useful when you only want specific sections of a long legal document.
        """
        success = False

        # Get the document fragments that match our search query
        jsonstr = self.fetch_doc_fragment(docid, q)
        if not jsonstr:
            return False

        # Create a filename that includes both the document ID and the search query
        jsonpath = self.storage.get_json_path('%d q: %s' % (docid, q))
        success = self.storage.save_json(jsonstr, jsonpath)
        return success    

    def download_doc(self, docid, docpath):    
        """
        Download a complete legal document and optionally its original PDF file.
        This is the main function for downloading court judgments and legal documents.
        """
        success = False
        orig_needed = self.orig  # Whether user wants the original PDF too
        jsonpath, origpath = self.storage.get_json_orig_path(docpath, docid)

        # Only download if we don't already have this document
        if not self.storage.exists(jsonpath):
            # Get the document data from the API
            jsonstr = self.fetch_doc(docid)

            try:
                # Parse the JSON response to check if it's valid
                d = json.loads(jsonstr)
            except Exception as e:
                self.logger.error('Error in getting doc %s %s', docid, e)
                return success

            # Check if the API returned an error message
            if 'errmsg' in d:
                self.logger.error('Error in getting doc %s', docid)
                return success
        
            # Successfully got the document - log it and save it
            self.logger.info('Saved %s', d['title'])
            self.storage.save_json(jsonstr, jsonpath)
            success = True

            # Check if we need the original PDF file
            if orig_needed:
                if not d['courtcopy']:  # If there's no original court copy available
                    orig_needed = False

        # Download the original PDF file if requested and available
        if orig_needed and not self.storage.exists_original(origpath):
            orig = self.fetch_orig_doc(docid)
            if orig and self.storage.save_original(orig, origpath):
                self.logger.info('Saved original %s', docid)
        return success        

    def make_query(self, q):
        """
        Build a complete search query by adding date filters and sorting options.
        This takes a basic search term and adds any date restrictions or sorting preferences.
        """
        # Add date range filters if specified
        if self.fromdate:
            q += ' fromdate: %s' % self.fromdate  # Only documents from this date onwards

        if self.todate:
            q += ' todate: %s' % self.todate      # Only documents up to this date

        # Add filter for documents added today
        if self.addedtoday:
            q += ' added:today'

        # Add sorting preference
        if self.sortby:
            q += ' sortby: ' + self.sortby        # Sort by newest or oldest first

        return q

    def download_doctype(self, doctype):
        """
        Download all documents of a specific type (e.g., all Supreme Court judgments).
        This searches for documents by their type and downloads all matching results.
        """
        # Create a search query for the specific document type
        q = 'doctypes: %s' % doctype
        q = self.make_query(q)

        pagenum = 0
        docids = []
        
        # Keep searching through pages until we get no more results
        while 1:
            results = self.search(q, pagenum, self.maxpages)
            obj = json.loads(results)
 
            # Stop if there are no more documents
            if 'docs' not in obj or len(obj['docs']) <= 0:
                break
                
            docs = obj['docs']
            self.logger.warning('Num results: %d, pagenum: %d', len(docs), pagenum)
            
            # Download each document found on this page
            for doc in docs:
                docpath = self.storage.get_docpath(doc['docsource'], doc['publishdate'])
                if self.download_doc(doc['tid'], docpath):
                    docids.append(doc['tid'])

            # Move to the next page of results
            pagenum += self.maxpages 

        return docids

    def save_search_results(self, q):
        """
        Search for documents matching a query and download all results.
        This creates a table of contents (CSV file) and downloads all matching documents.
        """
        # Create a directory for this search
        datadir = self.storage.get_search_path(q)

        # Create a CSV file to track all the documents we find
        tocwriter = self.storage.get_tocwriter(datadir)

        pagenum = 0
        current = 1
        docids  = []
        
        # Keep searching through pages until we get no more results
        while 1:
            results = self.search(q, pagenum, self.maxpages)
            obj = json.loads(results)

            # Check for errors from the API
            if 'errmsg' in obj:
                self.logger.warning('Error: %s, pagenum: %d q: %s', obj['errmsg'], pagenum, q)
                break

            # Stop if there are no more documents
            if 'docs' not in obj or len(obj['docs']) <= 0:
                break
            docs = obj['docs']
            if len(docs) <= 0:
                break
                
            # Log how many documents we found on this page
            self.logger.warning('Num results: %d, pagenum: %d found: %s q: %s', len(docs), pagenum, obj['found'], q)
            
            # Process each document found on this page
            for doc in docs:
                docid   = doc['tid']
                title   = doc['title']

                # Add this document to our table of contents
                toc = {'docid': docid, 'title': title, 'position': current, \
                       'date': doc['publishdate'], 'court': doc['docsource']}
                tocwriter.writerow(toc)

                # Choose how to organize the downloaded files
                if self.pathbysrc:
                    # Organize by court source and date
                    docpath = self.storage.get_docpath(doc['docsource'], doc['publishdate'])
                else:    
                    # Organize by position in search results
                    docpath = self.storage.get_docpath_by_position(datadir, current)
                    
                # Download the document
                if self.download_doc(docid, docpath):
                    docids.append(docid)
                current += 1

            # Move to the next page of results
            pagenum += self.maxpages 
        return docids

    def worker(self):
        """
        Worker function that runs in a separate process to handle search queries.
        This allows multiple searches to run at the same time for faster downloads.
        """
        while True:
            # Get the next search query from the queue
            q = self.queue.get()
            if q == None:
                # If we get None, it means we should stop working
                break

            self.logger.info('Processing %s', q)

            # Process this search query
            self.save_search_results(q)

            self.logger.info('Done with query %s', q)

    def execute_tasks(self, queries):
        """
        Run multiple search queries in parallel using multiple worker processes.
        This is much faster than running searches one after another.
        """
        workers = []
        
        # Start multiple worker processes
        for i in range(0, self.num_workers):
            process =  multiprocessing.Process(target = self.worker)
            process.start()
            workers.append(process)
      
        # Add all queries to the work queue
        for q in queries:
            q = self.make_query(q)
            self.queue.put(q)

        # Tell all workers to stop when they're done
        for process in workers:
            self.queue.put(None)

        # Wait for all workers to finish
        for process in workers:
            process.join()

# =============================================================================
# UTILITY FUNCTIONS - Helper functions used throughout the program
# =============================================================================

def get_dateobj(datestr):
    """
    Convert a date string (like "2023-12-25") into a Python date object.
    This helps organize files by date.
    """
    ds = re.findall('\\d+', datestr)  # Extract all numbers from the date string
    return datetime.date(int(ds[0]), int(ds[1]), int(ds[2]))  # Create date object

def mk_dir(datadir):
    """
    Create a directory if it doesn't already exist.
    This ensures we have a place to save downloaded files.
    """
    if not os.path.exists(datadir):
        os.mkdir(datadir)

# =============================================================================
# FILESTORAGE CLASS - Handles saving and organizing downloaded files
# =============================================================================
# This class manages where files are saved and how they're organized on your computer.
# It creates folders by court, date, and document type to keep everything organized.
# =============================================================================
class FileStorage:
    def __init__(self, datadir):
        """
        Initialize the file storage system with a base directory.
        All downloaded files will be saved under this directory.
        """
        self.datadir = datadir

    def save_json(self, results, filepath):
        """
        Save JSON data (like document text) to a file.
        This saves the main document content in a readable format.
        """
        json_doc  = results
        json_file = codecs.open(filepath, mode = 'w', encoding = 'utf-8')
        json_file.write(json_doc)
        json_file.close()

    def exists(self, filepath):
        """
        Check if a file already exists on disk.
        This prevents downloading the same document twice.
        """
        if os.path.exists(filepath):
            return True
        else:
            return False

    def exists_original(self, origpath):
        """
        Check if an original document file (like PDF) already exists.
        Original files can have different extensions (.pdf, .html, etc.)
        """
        return glob.glob('%s.*' % origpath)

    def get_docpath(self, docsource, publishdate):
        """
        Create a folder path organized by court source and date.
        Example: /data/Supreme Court/2023/2023-12-25/
        This keeps files organized and easy to find.
        """
        # Create folder for the court source (e.g., "Supreme Court")
        datadir = os.path.join(self.datadir, docsource)
        mk_dir(datadir)

        # Create subfolder for the year
        d = get_dateobj(publishdate)
        datadir = os.path.join(datadir, '%d' % d.year)
        mk_dir(datadir)

        # Create subfolder for the specific date
        docpath = os.path.join(datadir, '%s' % d)
        mk_dir(docpath)

        return docpath

    def get_file_extension(self, mtype):
        """
        Determine the file extension based on the content type.
        This helps save original documents with the correct file extension.
        """
        t = 'unkwn'  # Default to unknown
        if not mtype:
            pass 
        elif re.match('text/html', mtype):
            t = 'html'
        elif re.match('application/postscript', mtype):
            t = 'ps'
        elif re.match('application/pdf', mtype):
            t = 'pdf'
        elif re.match('text/plain', mtype):
            t = 'txt'
        elif re.match('image/png', mtype):
            t = 'png'
        return t 

    def save_original(self, orig, origpath):
        """
        Save an original document file (like a PDF) to disk.
        The document data comes encoded from the API and needs to be decoded.
        """
        obj = json.loads(orig)
        if 'errmsg' in obj:
            return False

        # Decode the base64-encoded document data
        doc = base64.b64decode(obj['doc'])

        # Determine the correct file extension
        extension = self.get_file_extension(obj['Content-Type'])

        # Save the file with the correct extension
        filepath   = origpath + '.%s' % extension
        filehandle = open(filepath, 'wb')
        filehandle.write(doc)
        filehandle.close()
        return True

    def get_docpath_by_docid(self, docid):
        """
        Create a simple folder path using just the document ID.
        Example: /data/12345/
        """
        docpath = os.path.join(self.datadir, '%d' % docid)
        mk_dir(docpath)
        return docpath

    def get_json_orig_path(self, docpath, docid):
        """
        Get the file paths for both the JSON document and original file.
        Returns two paths: one for the JSON text, one for the original PDF/image.
        """
        jsonpath = os.path.join(docpath, '%d.json' % docid)      # For document text
        origpath = os.path.join(docpath, '%d_original' % docid)  # For original file
        return jsonpath, origpath

    def get_json_path(self, q):
        """
        Create a file path for saving search results or document fragments.
        The filename includes the search query for easy identification.
        """
        jsonpath = os.path.join(self.datadir, '%s.json' % q)
        return jsonpath

    def get_search_path(self, q):
        """
        Create a directory for a specific search query.
        All documents from this search will be saved in this folder.
        """
        datadir = os.path.join(self.datadir, q)
        mk_dir(datadir)
        return datadir

    def get_tocwriter(self, datadir):
        """
        Create a CSV file to track all documents found in a search.
        This creates a table of contents with document titles, dates, courts, etc.
        """
        fieldnames = ['position', 'docid', 'date', 'court', 'title']
        tocfile   = os.path.join(datadir, 'toc.csv')
        tochandle = open(tocfile, 'w', encoding = 'utf8')
        tocwriter = csv.DictWriter(tochandle, fieldnames=fieldnames)
        tocwriter.writeheader()
        return tocwriter

    def get_docpath_by_position(self, datadir, current):
        """
        Create a folder path organized by position in search results.
        Example: /data/search_query/1/, /data/search_query/2/, etc.
        """
        docpath = os.path.join(datadir, '%d' % current)
        mk_dir(docpath)
        return docpath

# =============================================================================
# COMMAND LINE ARGUMENT PARSER - Handles user input and options
# =============================================================================
# This function sets up all the command-line options that users can specify
# when running the program. It's like the settings menu for the downloader.
# =============================================================================

def get_arg_parser():
    """
    Create the command-line argument parser with all available options.
    This defines what options users can specify when running the program.
    """
    parser = argparse.ArgumentParser(description='For downloading from the api.indiankanoon.org endpoint', add_help=True)
    
    # Logging options - control how much information is shown
    parser.add_argument('-l', '--loglevel', dest='loglevel', action='store',\
                        required = False, default = 'info', \
                        help='log level(error|warning|info|debug)')

    parser.add_argument('-g', '--logfile', dest='logfile', action='store',\
                        required = False, default = None, help='log file')
   
    # Search and filtering options
    parser.add_argument('-c', '--doctype', dest='doctype', action='store',\
                        required= False, help='doctype')
    parser.add_argument('-f', '--fromdate', dest='fromdate', action='store',\
                        required= False, help='from date in DD-MM-YYYY format')
    parser.add_argument('-t', '--todate', dest='todate', action='store',\
                        required= False, help='to date in DD-MM-YYYY format')
    parser.add_argument('-S', '--sortby', dest='sortby', action='store',\
                        required= False, help='sort results by (mostrecent|leastrecent)')

    # Required options - these must be specified
    parser.add_argument('-D', '--datadir', dest='datadir', action='store',\
                        required= True,help='directory to store files')
    parser.add_argument('-s', '--sharedtoken', dest='token', action='store',\
                        required= True,help='api.ik shared token')

    # What to download - choose one of these options
    parser.add_argument('-q', '--query', dest='q', action='store',\
                        required = False, help='ik query')
    parser.add_argument('-Q', '--qfile', dest='qfile', action='store',\
                        required = False, help='queries in a file')
    parser.add_argument('-d', '--docid', type = int, dest='docid', \
                        action='store', required = False, help='ik docid')

    # Download options
    parser.add_argument('-o', '--original', dest='orig', action='store_true',\
                        required = False,   help='ik original')

    # Citation and pagination limits
    parser.add_argument('-m', '--maxcites', type = int, dest='maxcites', \
                        action='store', default = 0, required = False, \
                        help='doc maxcites')
    parser.add_argument('-M', '--maxcitedby', type = int, dest='maxcitedby', \
                        action='store', default = 0, required = False, \
                        help='doc maxcitedby')
    parser.add_argument('-p', '--maxpages', type = int, dest='maxpages', \
                        action='store', required = False, \
                        help='max search result pages', default=1)
    parser.add_argument('-P', '--pathbysrc', dest='pathbysrc', \
                        action='store_true', required = False, \
                        help='save docs by src')
    parser.add_argument('-a', '--addedtoday', dest='addedtoday', \
                        action='store_true', required = False, default = False,\
                        help='Search only for documents that were added today')
    parser.add_argument('-N', '--workers', type = int, dest='numworkers', \
                        action='store', default = 5, required = False, \
                        help='num workers for parallel downloads')
    return parser

# =============================================================================
# LOGGING SETUP - Controls how the program reports what it's doing
# =============================================================================
# These functions set up logging so users can see what the program is doing
# and troubleshoot any problems that occur.
# =============================================================================

# Define how log messages should be formatted
logformat   = '%(asctime)s: %(name)s: %(levelname)s %(message)s'
dateformat  = '%Y-%m-%d %H:%M:%S'

def initialize_file_logging(loglevel, filepath):
    """
    Set up logging to write messages to a file.
    This is useful for keeping a record of what happened during downloads.
    """
    logging.basicConfig(\
        level    = loglevel,   \
        format   = logformat,  \
        datefmt  = dateformat, \
        stream   = filepath
    )

def initialize_stream_logging(loglevel = logging.INFO):
    """
    Set up logging to display messages on the screen.
    This shows progress and any errors while the program is running.
    """
    logging.basicConfig(\
        level    = loglevel,  \
        format   = logformat, \
        datefmt  = dateformat \
    )

def setup_logging(level, filename = None):
    """
    Set up logging based on user preferences.
    Can log to a file, to the screen, or both.
    """
    # Convert text level names to logging constants
    leveldict = {'critical': logging.CRITICAL, 'error': logging.ERROR, \
                 'warning': logging.WARNING,   'info': logging.INFO, \
                 'debug': logging.DEBUG}
    loglevel = leveldict[level]

    if filename:
        # Log to a file
        filestream = codecs.open(filename, 'w', encoding='utf8')
        initialize_file_logging(loglevel, filestream)
    else:
        # Log to the screen
        initialize_stream_logging(loglevel)


# =============================================================================
# MAIN PROGRAM - This is where everything starts when you run the script
# =============================================================================
# This section handles command-line arguments and decides what to download
# based on what the user specified.
# =============================================================================

if __name__ == '__main__':
    # Parse command-line arguments to understand what the user wants to do
    parser = get_arg_parser()
    args   = parser.parse_args()

    # Set up logging so we can see what's happening
    setup_logging(args.loglevel, filename = args.logfile)

    # Create a logger for this program
    logger = logging.getLogger('ikapi')

    # Create the file storage system and API client
    filestorage = FileStorage(args.datadir) 
    ikapi       = IKApi(args, filestorage)

    # Decide what to do based on the command-line arguments
    if args.docid != None and args.q:
        # Download specific parts of a document that match a search query
        logger.warning('Docfragment for %d q: %s', args.docid, args.q)
        ikapi.save_doc_fragment(args.docid, args.q)
    elif args.docid != None:
        # Download a specific document by its ID number
        ikapi.download_doc(args.docid, args.datadir)
    elif args.q:
        # Search for documents matching a query and download all results
        q = args.q
        if args.addedtoday:
            q += ' added:today'
        logger.warning('Search q: %s', q)
        ikapi.save_search_results(q)
    elif args.doctype:
        # Download all documents of a specific type (e.g., all Supreme Court judgments)
        ikapi.download_doctype(args.doctype)
    elif args.qfile:
        # Process multiple search queries from a file
        queries = []
        filehandle = open(args.qfile, 'r', encoding='utf8')
        for line in filehandle.readlines():
            queries.append(line.strip())
        ikapi.execute_tasks(queries)
        filehandle.close()
