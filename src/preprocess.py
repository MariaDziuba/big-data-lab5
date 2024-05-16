from pyspark.ml.feature import VectorAssembler, StandardScaler
import loguru


class Preprocessor:

    def load_dataset(self, path_to_data, spark):
        dataset = spark.read.csv(
            path_to_data,
            header=True,
            inferSchema=True,
            sep='\t',
        )
        dataset.fillna(value=0)
        loguru.logger.info("Filled all NaN values with zeroes")

        output_col = 'features'
        vector_assembler = VectorAssembler(
            inputCols=[
                # 'last_updated_t',
                'completeness',
                # 'last_image_t',
                'energy-kcal_100g',
                'energy_100g',
                'fat_100g',
                'saturated-fat_100g',
                'carbohydrates_100g',
                'sugars_100g',
                'proteins_100g',
                'salt_100g',
                'sodium_100g'
            ],
            outputCol=output_col,
            handleInvalid='skip',
        )

        assembled_data = vector_assembler.transform(dataset)
        loguru.logger.info("The dataset was assembled")

        return assembled_data


    def scale_assembled_dataset(self, assembled_data):
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features'
        )
        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)
        loguru.logger.info("The dataset was scaled using StandardScaler")

        return scaled_data