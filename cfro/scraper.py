# This library is used to parse the Google News API RSS output.
# https://pypi.org/project/feedparser/
# https://pythonhosted.org/feedparser/introduction.html
import feedparser
import time
from datetime import datetime, timedelta

# This library is used to run multithreaded downloads.
# https://docs.python.org/3/library/concurrent.futures.html
import concurrent.futures

# This library is used to parse news articles and extract images.
# https://newspaper.readthedocs.io/en/latest/
# https://pypi.org/project/newspaper3k/
import newspaper

# This library is used to set timeouts on download requests.
# https://docs.python.org/3/library/socket.html
import socket

# These libraries are used to download images and save to OS.
import urllib.request
import os

# This is used to detect and remove duplicate images.
from difPy import dif
import networkx as nx
import shutil

from .dataset import Photo

from enum import Enum


class ImageSource(Enum):
    GOOGLE_NEWS = 1


class ImageScraper:
    """
    This is a wrapper for downloading images from news
    articles and other sources.

    Images are downloaded to an input folder then renamed as
    000000.jpg, 000001.jpg, 000002.jpg, ...

    The main entrypoint is `download_images()`.

    References:
    * https://stackoverflow.com/a/7829688
    * https://medium.com/rakuten-rapidapi/top-10-best-news-apis-google-news-bloomberg-bing-news-and-more-bbf3e6e46af6
    * http://theautomatic.net/2020/08/05/how-to-scrape-news-articles-with-python/
    * https://stackoverflow.com/questions/3042757/downloading-a-picture-via-urllib-and-python
    """

    # Timeout on a single image download.
    TIMEOUT = 5

    NUM_WORKERS = 5

    def _passes_pub_date_filter(rss_entry, max_days_old):
        cutoff = datetime.now() - timedelta(days=max_days_old)
        pub_date = datetime.fromtimestamp(time.mktime(rss_entry["published_parsed"]))
        # if pub_date <= cutoff:
        #     print(f"Article {rss_entry['link']} fails the {max_days_old} day cutoff ({pub_date} <= {cutoff})")
        return pub_date > cutoff

    def _gen_google_news_article_urls(query, max_days_old=None):
        """
        Returns a list of urls, search results of `query` on Google News.
        """
        query = query.replace(" ", "+")
        rss_link = f"http://news.google.com/news?q={query}&output=rss"
        downloaded_rss = feedparser.parse(rss_link)
        entries = downloaded_rss["entries"]

        # TODO - see if the news API can return 100 articles from the past N days.
        # (https://newscatcherapi.com/blog/google-news-rss-search-parameters-the-missing-documentaiton).
        # In contrast, our method has the news API return 100 articles from any date
        # and then filters down from 100. The suggested fix could increase dataset size.

        if max_days_old is not None:
            orig_num_entries = len(entries)
            entries = [
                entry
                for entry in entries
                if ImageScraper._passes_pub_date_filter(entry, max_days_old)
            ]
            print(
                "Filtering by date eliminated",
                orig_num_entries - len(entries),
                "articles.",
            )

        return [entry["link"] for entry in entries]

    def _gen_image_urls_from_article(article_url):
        """
        Returns a list of image urls found on `article_url`.
        If the article is invalid or no images are found,
            an empty list is returned.
        """
        try:
            article = newspaper.Article(article_url)
            article.download()
            # Ignore the encoding error in `article.parse()` (possibly from running
            # Python 3.9 instead of Python 3.10). It goes away when `article.parse()`
            # is commented out and [] is returned and only affects < 10 articles per person. Example:
            # "encoding error : input conversion failed due to input error, bytes 0x21 0x00 0x00 0x00"
            article.parse()
            # To obtain more images, use `return article.images` here (probably not needed).
            top_image = article.top_image
            # Ensure the image exists and is not a gif
            valid_image = len(top_image) > 0 and ".gif" not in top_image
            print(".", end="", flush=True)
            return [top_image] if valid_image else []
        except:
            return []

    def _gen_image_urls_from_articles(article_urls):
        """
        Returns image urls collected from articles in `article_urls`
        """
        image_urls = []
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ImageScraper.NUM_WORKERS
        ) as executor:
            # Start the parsing operations and mark each future with its URL
            futures = [
                executor.submit(ImageScraper._gen_image_urls_from_article, url)
                for url in article_urls
            ]
            for job in concurrent.futures.as_completed(futures):
                curr_image_urls = job.result()
                image_urls.extend(curr_image_urls)
        return list(set(image_urls))

    def _gen_image_urls_for_person(person_name, download_source, max_days_old=None):
        """
        Returns a list of image urls based on a query of
        param `download_source` to find images of `person_name`
        """
        if download_source != ImageSource.GOOGLE_NEWS:
            raise NotImplementedError("Only Google News is supported.")

        article_urls = ImageScraper._gen_google_news_article_urls(
            person_name, max_days_old=max_days_old
        )
        image_urls = ImageScraper._gen_image_urls_from_articles(article_urls)
        return image_urls

    def _load_url(link, photo_dir, i):
        """
        Downloads an image from `link` to `photo_dir` with id `i`.
        Applies a "temp_" prefix.
        """
        filename = ImageScraper.get_image_filename(photo_dir, i, temp=True)
        try:
            urllib.request.urlretrieve(link, filename)
            print(".", end="", flush=True)
        except:
            # We could potentially do better error handling here.
            return None
        return (filename, link)

    def _download_images_helper(photo_dir, urls):
        """
        Download the images from `urls` to `photo_dir` with increasing
        id's starting from `min_photo_id`. The output filenames returned
        may not have contiguous ids due to failed download ops.
        """
        output_files = []

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ImageScraper.NUM_WORKERS
        ) as executor:
            # Start the load operations and mark each future with its URL
            futures = [
                executor.submit(ImageScraper._load_url, url, photo_dir, i)
                for i, url in enumerate(urls)
            ]
            for i in concurrent.futures.as_completed(futures):
                result = i.result()
                if result is not None:
                    output_files.append(result)

        return output_files

    def _run_difpy(duplicate_photo_dir):
        # Run the difPy library with the lowest similarity threshold
        # (i.e., will detect the greatest # of duplicate pairs).
        search = dif(duplicate_photo_dir, similarity="low")

        # Construct an undirected graph where edges
        # represent duplicate images per difPy.
        G = nx.Graph()
        for d in search.result.values():
            orig = d["location"]
            dups = d["duplicates"]
            for dup in dups:
                G.add_edge(orig, dup)

        # Compute the connected components and select the highest
        # quality image per component to keep in the dataset.
        orig_to_dups = {}
        cc = nx.connected_components(G)
        for c in cc:
            dups = set(c)
            # This method of using st_size for quality is used
            # originally in the difPy library.
            orig = max(c, key=lambda f: os.stat(f).st_size)
            dups.remove(orig)
            orig_to_dups[orig] = dups

        return orig_to_dups

    def _remove_duplicates(photo_dir, downloads, duplicate_photo_dir):
        init_num_downloads = len(downloads)

        # Create a temp folder in the duplicate dir.
        temp_duplicate_photo_dir = duplicate_photo_dir + os.sep + "temp"
        os.mkdir(temp_duplicate_photo_dir)

        # This is the next subdirectory idx where pairs of duplicates will go.
        num_duplicate_groups = len(os.listdir(duplicate_photo_dir))

        # Move all downloads into the duplicate_photo_dir temp subdirectory.
        for (temp_filename, _) in downloads:
            assert temp_filename.count(photo_dir) == 1
            os.rename(
                temp_filename,
                temp_filename.replace(photo_dir, temp_duplicate_photo_dir),
            )

        # Run the DIF module after cd-ing into the duplicate directory.
        orig_to_dups = ImageScraper._run_difpy(temp_duplicate_photo_dir)

        # Store the duplicate groups for post-analysis.
        all_dups = set()
        for orig, dups in orig_to_dups.items():
            all_dups.update(dups)
            sub_dir = duplicate_photo_dir + os.sep + str(num_duplicate_groups)
            num_duplicate_groups += 1
            os.mkdir(sub_dir)
            for i, image in enumerate(dups):
                os.rename(image, sub_dir + os.sep + f"dup{i}.jpg")
            # Only the highest quality image remains in the dataset.
            shutil.copyfile(orig, sub_dir + os.sep + "best_quality_dup.jpg")

        # Filter out the duplicates from the downloads list.
        all_dups = set([f.split(os.sep)[-1] for f in all_dups])
        non_downloads = [
            url
            for (temp_filename, url) in downloads
            if temp_filename.split(os.sep)[-1] in all_dups
        ]
        downloads = [
            (temp_filename, url)
            for (temp_filename, url) in downloads
            if temp_filename.split(os.sep)[-1] not in all_dups
        ]

        # Debugging - you can check these images don't appear in the dataset.
        # print('Not downloading the following duplicated images:', non_downloads)
        assert len(downloads) + len(all_dups) == init_num_downloads

        # Move the non-duplicate photos back to the actual photos dir.
        for (temp_filename, _) in downloads:
            os.rename(
                temp_filename.replace(photo_dir, temp_duplicate_photo_dir),
                temp_filename,
            )

        # Assert that we cleared the duplicate dir's temp folder.
        assert len(os.listdir(temp_duplicate_photo_dir)) == 0

        # Remove the temp folder.
        os.rmdir(temp_duplicate_photo_dir)

        return downloads

    def _parse_downloads(photo_dir, downloads, min_photo_id):
        """
        Renumber images so the downloaded images are contiguous from
        min_photo_id to min_photo_id + N - 1.
        """
        photos = []
        for i, (temp_filename, source_url) in enumerate(downloads):
            photo_id = min_photo_id + i
            new_filename = ImageScraper.get_image_filename(photo_dir, photo_id)
            os.rename(temp_filename, new_filename)
            # At this stage, the filename is final, so we can prepare
            # a Photo for the database.
            photos.append(Photo(photo_id, source_url))
        return photos

    def get_image_filename(photo_dir, i, temp=False):
        """
        Returns a filename in `photo_dir` for the image with
        id `i` (using up to 6 zero's for padding).
        """
        prefix = "temp_" if temp else ""
        return f"{photo_dir}{os.sep}{prefix}{i:06}.jpg"

    def download_images(
        person_name,
        photo_dir,
        download_source,
        min_photo_id,
        max_number_of_photos=None,
        duplicate_photo_dir=None,
        scrape_articles_from_past_n_days=None,
    ):
        """
        Returns a list of Photo's downloaded based on the config params.

        Params:
        * person_name (str) is a search keyword, e.g., 'Ethan Mann'
        * photo_dir (str) is an existing dir for downloaded images
        * download_source (enum) is the ImageSource to scrape from
        * min_photo_id (int) is the first available idx for new filenames
        * max_number_of_photos (optional int) is the max number of downloads
        * duplicate_photo_dir (optional str) determines if dups are removed
        * scrape_articles_from_past_n_days (optional int) filters by pub date
        """
        print(f"Scraping news articles for person {person_name}: ", end="", flush=True)

        urls = ImageScraper._gen_image_urls_for_person(
            person_name, download_source, max_days_old=scrape_articles_from_past_n_days
        )

        # This is helpful for debugging
        if max_number_of_photos is not None and len(urls) > max_number_of_photos:
            # print(f"Urls truncated from {len(urls)} to {max_number_of_photos}")
            urls = urls[:max_number_of_photos]
            assert len(urls) == max_number_of_photos

        print()

        socket.setdefaulttimeout(ImageScraper.TIMEOUT)

        print(f"Downloading images for person {person_name}: ", end="", flush=True)

        # Download images we scraped from news articles.
        downloads = ImageScraper._download_images_helper(photo_dir, urls)

        print()

        # Restore the default timeout for downloads.
        socket.setdefaulttimeout(None)

        # Remove duplicates
        if duplicate_photo_dir is not None:
            downloads = ImageScraper._remove_duplicates(
                photo_dir, downloads, duplicate_photo_dir
            )

        # Assign actual id's to each successful download.
        photos = ImageScraper._parse_downloads(photo_dir, downloads, min_photo_id)
        return photos
