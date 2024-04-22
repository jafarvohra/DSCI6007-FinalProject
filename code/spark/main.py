from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession, functions as F
from pyspark.sql.types import *
import re
import boto3
from warcio.archiveiterator import ArchiveIterator
from io import BytesIO
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
import argparse


def get_command_line_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", "-in", type=str, default="", help="Input file name"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, default="", help="Output file path"
    )
    return vars(parser.parse_args())


def html_to_text(page):
    """Convert an HTML page to text."""
    try:
        encoding = EncodingDetector.find_declared_encoding(page, is_html=True)
        soup = BeautifulSoup(page, "lxml", from_encoding=encoding)
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(" ", strip=True)
    except Exception as e:
        return ""


def fetch_process_warc_records(rows):
    """Retrieve documents from S3 and process them."""
    s3client = boto3.client("s3")
    for row in rows:
        warc_path = row["warc_filename"]
        offset = int(row["warc_record_offset"])
        length = int(row["warc_record_length"])
        range_request = f"bytes={offset}-{offset + length - 1}"
        response = s3client.get_object(
            Bucket="commoncrawl", Key=warc_path, Range=range_request
        )
        record_stream = BytesIO(response["Body"].read())
        for record in ArchiveIterator(record_stream):
            page = record.content_stream().read()
            text = html_to_text(page)
            word_pattern = re.compile(r"\$+", re.UNICODE)
            iterator = word_pattern.finditer(text)
            price_indices = [match.span() for match in iterator]
            sentences = []
            for start, end in price_indices:
                if start - 300 < 0:
                    sentences.append(text[0 : end + 10])
                else:
                    sentences.append(text[start - 300 : end + 10])
            for sentence in sentences:
                title = None
                price = None
                if sentence.find("$") != -1:
                    price = sentence[sentence.find("$") :]
                    if sentence.find("Title") != -1 and (sentence.find("$") - 100) > 0:
                        title = sentence[
                            sentence.find("Title") + 6 : sentence.find("$") - 100
                        ]
                    yield title, price
                else:
                    yield title, price


def main():
    args = get_command_line_args()
    conf = SparkConf().setAll(
        [
            ("spark.executor.memory", "10g"),
            ("spark.executor.instances", "8"),
            ("spark.executor.cores", "3"),
            ("spark.dynamicAllocation.enabled", "true"),
        ]
    )

    session = SparkSession.builder.master("yarn").config(conf=conf).getOrCreate()

    input_path = f"s3://athena-east-1-satish/{args['input_file']}"
    sqldf = (
        session.read.format("parquet").option("header", True).load(input_path + "/*")
    )

    warc_recs = sqldf.select(
        "warc_filename", "warc_record_offset", "warc_record_length"
    ).rdd.repartition(40)

    products = warc_recs.mapPartitions(fetch_process_warc_records)
    sqlContext = SQLContext(session)

    schema_product = sqlContext.createDataFrame(products, ["Product", "Price"])

    output_path = f"s3://emr-output-bucket-unh/{args['output_path']}/"
    schema_product.write.parquet(output_path)


if __name__ == "__main__":
    main()
