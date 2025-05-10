from typing import Union, Optional, List
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame

class Dataset:
    """
    Class for creating Dataset for handling data for different task for lib
    """
    def __init__(
        self,
        data_source: Union[pd.DataFrame, pl.DataFrame, SparkDataFrame, str],
        target_col: str,
        id_cols: List[str] = None,
        backend: Optional[str] = None,
    ):

        self.target_col = target_col
        self.id_cols = id_cols

        if isinstance(data_source, (pd.DataFrame, pl.DataFrame, SparkDataFrame)):
            self.data = data_source
            self.backend = ("pandas" if isinstance(data_source, pd.DataFrame) else
                           "polars" if isinstance(data_source, pl.DataFrame) else
                           "spark")
        elif isinstance(data_source, str):
            self._load_from_file(data_source, backend)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

    def show(self, n: int = 5):
        if self.backend == "pandas":
            return self.df.head(n)
        elif self.backend == "polars":
            return self.df.head(n).to_pandas()
        elif self.backend == "spark":
            return self.df.show(n)

    def get_column_names(self):
        if self.backend == "pandas":
            return self.df.columns.tolist()
        elif self.backend == "polars":
            return self.df.columns
        elif self.backend == "spark":
            return self.df.columns

    def filter_data(self, condition: str):
        if self.backend == "pandas":
            return self.df.query(condition)
        elif self.backend == "polars":
            return self.df.filter(pl.col(condition))
        elif self.backend == "spark":
            return self.df.filter(condition)

    def convert_to_pandas(self):
        if self.backend == "polars":
            return self.df.to_pandas()
        elif self.backend == "spark":
            return self.df.toPandas()
        else:
            return self.df

    def convert_to_polars(self):
        if self.backend == "pandas":
            return pl.from_pandas(self.df)
        elif self.backend == "spark":
            return pl.from_pandas(self.df.toPandas())
        else:
            return self.df

    def convert_to_spark(self):
        if self.backend == "pandas":
            spark = SparkSession.builder.getOrCreate()
            return spark.createDataFrame(self.df)
        elif self.backend == "polars":
            spark = SparkSession.builder.getOrCreate()
            return spark.createDataFrame(self.df.to_pandas())
        else:
            return self.df
