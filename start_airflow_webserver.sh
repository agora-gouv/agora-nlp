./install_airflow.sh
airflow db upgrade
airflow db migrate
airflow webserver