import os
import logging
import pytest
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering,\
	classification_report_image, feature_importance_plot, train_models
PATH  = r"data\bank_data.csv"


logging.basicConfig(filename=r'logs\churn_library.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def test_import_data():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(PATH)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_perform_eda():
	'''
	test perform eda function
	'''
	path_data = Path(PATH)
	df = import_data(path_data)
	EDA_PATH = "images\eda"

	perform_eda(df)
	correlation = Path(EDA_PATH, "Correlation.png")
	churn_rate = Path(EDA_PATH, "churn_rate.png")
	customer_age = Path(EDA_PATH, "customer_age.png")
	Marital_status = Path(EDA_PATH, "Marital_status.png")
	Total_Trans_Ct = Path(EDA_PATH, "Total_Trans_Ct.png")


	assert correlation.exists()
	assert churn_rate.exists()
	assert customer_age.exists()
	assert Marital_status.exists()
	assert Total_Trans_Ct.exists()




def test_encoder_helper():
	'''
	test encoder helper
	'''
	df = import_data(PATH)
	df = encoder_helper(df)
	
	for x in df.columns:
		try :
			assert is_numeric_dtype(df[x])
			logging.info(f"colonne {x} est bien numerique")
		except ValueError as err:
			logging.error(f"La colonne {x} n'est pas au format numérique")



def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	df = import_data(PATH)
	df = test_encoder_helper(df)
	perform_feature_engineering(dataset)



# def test_train_models(train_models):
# 	pass
# 	'''
# 	test train_models
# 	'''


if __name__ == "__main__":
	test_import_data()








