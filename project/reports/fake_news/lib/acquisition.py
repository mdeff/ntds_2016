import configparser
import dateutil
import datetime
import requests

import pandas as pd


class Feature(object):
    """
    A feature represent a Facebook graph API field in addition to a
    customized name and a formatting function used to clean the collected data
    """

    def __init__(self, fbquery, formatter=None, name=None):
        """Initialize the feature

        Arguments:
        ----------
        fbquery : str
            Field of the Facebook Graph API to query
        formatter : function (default: None)
            Function taking as input the data returned by the API query and
            returning the formatted the data, if `None` then no formatting will
            be performed and the raw data will be returned
        name : str (default: None)
            Name of the feature used in our system, if `None` then the value of
            `fbquery` will be used
        """
        self.fbquery = fbquery
        self.format = formatter or (lambda x: x.get(fbquery, ''))
        self.name = name or fbquery


class FacebookScraper(object):
    """
    A web scraper for the Facebook Graph API that take care of the data
    cleaning process and format data in a Pandas Dataframe
    """

    URL_TEMPLATE = ('https://graph.facebook.com/v2.8/{page}/posts?'
        'fields={fields}&since={since}&until={until}&access_token={token}')

    def __init__(self, field_list):
        """
        Initialize the scrapper with a list of fields

        Arguments:
        ----------
        field_list : array-like
            Iterable of Feature objects containing the features to scrape
        """
        self.field_list = field_list
        self.data = None

    def _build_field_query(self):
        """
        Prepare the field query string for the API
        """
        return ','.join([f.fbquery for f in self.field_list])

    def _build_column_list(self):
        """
        Prepare the list of columns for the Pandas Dataframe
        """
        return ['page'] + [f.name for f in self.field_list]

    def extract_token(self, credential_file='credentials.ini'):
        """
        Read the confidential token
        """
        credentials = configparser.ConfigParser()
        credentials.read(credential_file)
        self.token = credentials.get('facebook', 'token')

    def initialize_dataframe(self, overwrite=False):
        """
        Initialize the dataframe if  it does not exist yet and ovewrite it if
        necessary
        """
        if (self.data is None) or overwrite:
            columns = self._build_column_list()
            self.data = pd.DataFrame(columns=columns)

    def run(self, page, since='2016/01/01', until='today', overwrite=False):
        """
        Scrape the page since date `since` until date `until` (a unix
        timestamp or any date accepted by strtotime). An exception is raised if
        the Facebook Graph API returns an error. Returns the data in a well
        formatted Pandas Dataframe.

        Arguments:
        ----------
        page : str
            Facebook page name
        since : str
            Start scraping the page at date `since`
        until : str
            Stop scraping the page at date `until`
        overwrite : bool (default: False)
            Wether or not to overwrite the existing Pandas Dataframe, if
            `False` new data are appended to the existing dataframe, if `True`
            existing data are overwritten and a new dataframe is intialized
        """

        # Initialize the dataframe if necessary (or overwrite it)
        self.initialize_dataframe(overwrite)

        # Build the initial request url
        field_query = self._build_field_query()
        url = self.URL_TEMPLATE.format(page=page, fields=field_query,
                                       since=since, until=until,
                                       token=self.token)

        # Query Facebook using pagination
        while True:
            # Get the data
            posts = requests.get(url).json()

            # Stop if an error occured
            if 'error' in posts:
                print(posts)
                print(url)
                raise Exception('Facebook API Error: {}'.format(posts['error']['message']))

            # Extract information for each of the received post.
            for post in posts['data']:
                # Clean the raw post
                serie = {f.name: f.format(post) for f in self.field_list}
                serie['page'] = page
                # Add the dictionary as a new line to the pandas DataFrame.
                self.data = self.data.append(serie, ignore_index=True)

            try:
                # Get the url of the next page
                url = posts['paging']['next']
            except KeyError:
                # No more posts.
                break

        return self.data
