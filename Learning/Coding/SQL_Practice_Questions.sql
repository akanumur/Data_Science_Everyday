#SQL Script to Seed Sample Data
CREATE DATABASE ORG;
SHOW DATABASES;
USE ORG;

CREATE TABLE Worker (
	WORKER_ID INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
	FIRST_NAME CHAR(25),
	LAST_NAME CHAR(25),
	SALARY INT(15),
	JOINING_DATE DATETIME,
	DEPARTMENT CHAR(25)
);

INSERT INTO Worker 
	(WORKER_ID, FIRST_NAME, LAST_NAME, SALARY, JOINING_DATE, DEPARTMENT) VALUES
		(001, 'Monika', 'Arora', 100000, '14-02-20 09.00.00', 'HR'),
		(002, 'Niharika', 'Verma', 80000, '14-06-11 09.00.00', 'Admin'),
		(003, 'Vishal', 'Singhal', 300000, '14-02-20 09.00.00', 'HR'),
		(004, 'Amitabh', 'Singh', 500000, '14-02-20 09.00.00', 'Admin'),
		(005, 'Vivek', 'Bhati', 500000, '14-06-11 09.00.00', 'Admin'),
		(006, 'Vipul', 'Diwan', 200000, '14-06-11 09.00.00', 'Account'),
		(007, 'Satish', 'Kumar', 75000, '14-01-20 09.00.00', 'Account'),
		(008, 'Geetika', 'Chauhan', 90000, '14-04-11 09.00.00', 'Admin');

CREATE TABLE Bonus (
	WORKER_REF_ID INT,
	BONUS_AMOUNT INT(10),
	BONUS_DATE DATETIME,
	FOREIGN KEY (WORKER_REF_ID)
		REFERENCES Worker(WORKER_ID)
        ON DELETE CASCADE
);

INSERT INTO Bonus 
	(WORKER_REF_ID, BONUS_AMOUNT, BONUS_DATE) VALUES
		(001, 5000, '16-02-20'),
		(002, 3000, '16-06-11'),
		(003, 4000, '16-02-20'),
		(001, 4500, '16-02-20'),
		(002, 3500, '16-06-11');
        
CREATE TABLE Title (
	WORKER_REF_ID INT,
	WORKER_TITLE CHAR(25),
	AFFECTED_FROM DATETIME,
	FOREIGN KEY (WORKER_REF_ID)
		REFERENCES Worker(WORKER_ID)
        ON DELETE CASCADE
);

INSERT INTO Title 
	(WORKER_REF_ID, WORKER_TITLE, AFFECTED_FROM) VALUES
 (001, 'Manager', '2016-02-20 00:00:00'),
 (002, 'Executive', '2016-06-11 00:00:00'),
 (008, 'Executive', '2016-06-11 00:00:00'),
 (005, 'Manager', '2016-06-11 00:00:00'),
 (004, 'Asst. Manager', '2016-06-11 00:00:00'),
 (007, 'Executive', '2016-06-11 00:00:00'),
 (006, 'Lead', '2016-06-11 00:00:00'),
 (003, 'Lead', '2016-06-11 00:00:00');
 
#Q-1. Write an SQL query to fetch “FIRST_NAME” from Worker table using the alias name as <WORKER_NAME>
SELECT FIRST_NAME AS WORKER_NAME FROM Worker;
 
#Q-2. Write an SQL query to fetch “FIRST_NAME” from Worker table in upper case.
SELECT upper(FIRST_NAME) FROM WORKER;
 
#Q-3. Write an SQL query to fetch unique values of DEPARTMENT from Worker table.
SELECT DISTINCT(DEPARTMENT) FROM WORKER;
 
#Q-4. Write an SQL query to print the first three characters of  FIRST_NAME from Worker table.
SELECT LEFT(FIRST_NAME,3) FROM WORKER; #ALTERNATIVELY WE CAN USE SUBSTRING FUNCTION
 
#Q-5. Write an SQL query to find the position of the alphabet (‘a’) in the first name column ‘Amitabh’ from Worker table.
#INSTR function returns the position of a substring in a string
#BY USING BINARY WE ARE MAKING 'A' CASE SENSITIVE
SELECT INSTR(FIRST_NAME,BINARY'a') FROM WORKER WHERE FIRST_NAME = 'Amitabh'; 
 
#Q-6. Write an SQL query to print the FIRST_NAME from Worker table after removing white spaces from the right side
SELECT RTRIM(FIRST_NAME) FROM WORKER;
 
#Q-7. Write an SQL query to print the DEPARTMENT from Worker table after removing white spaces from the left side.
SELECT LTRIM(DEPARTMENT) FROM WORKER;
 
#Q-8. Write an SQL query that fetches the unique values of DEPARTMENT from Worker table and prints its length.
SELECT DISTINCT LENGTH(DEPARTMENT) FROM WORKER;
 
#Q-9. Write an SQL query to print the FIRST_NAME from Worker table after replacing ‘a’ with ‘A’.
SELECT REPLACE(FIRST_NAME,'a','A') FROM WORKER;
 
#Q-10. Write an SQL query to print the FIRST_NAME and LAST_NAME from Worker table into a single column COMPLETE_NAME. A space char should separate them.
SELECT CONCAT(FIRST_NAME , " " , LAST_NAME) AS COMPLETE_NAME FROM WORKER;
 
#Q-11. Write an SQL query to print all Worker details from the Worker table order by FIRST_NAME Ascending
SELECT * FROM WORKER ORDER BY FIRST_NAME;
 
#Q-12. Write an SQL query to print all Worker details from the Worker table order by FIRST_NAME Ascending and DEPARTMENT Descending.
SELECT * FROM WORKER ORDER BY FIRST_NAME ASC,DEPARTMENT DESC;

#Q-13. Write an SQL query to print details for Workers with the first name as “Vipul” and “Satish” from Worker table.
SELECT * FROM WORKER WHERE FIRST_NAME IN ('VIPUL','SATISH');

#Q-14. Write an SQL query to print details of workers excluding first names, “Vipul” and “Satish” from Worker table.
SELECT * FROM WORKER WHERE FIRST_NAME NOT IN ('VIPUL','SATISH');

#Q-15. Write an SQL query to print details of Workers with DEPARTMENT name as “Admin”.
SELECT * FROM WORKER WHERE DEPARTMENT LIKE 'ADMIN';

#Q-16. Write an SQL query to print details of the Workers whose FIRST_NAME contains ‘a’.
SELECT * FROM WORKER WHERE FIRST_NAME LIKE '%A%';

#Q-17. Write an SQL query to print details of the Workers whose FIRST_NAME ends with ‘a’.
SELECT * FROM WORKER WHERE FIRST_NAME LIKE '%A';

#Q-18. Write an SQL query to print details of the Workers whose FIRST_NAME ends with ‘h’ and contains six alphabets.
SELECT * FROM WORKER WHERE FIRST_NAME LIKE '%h' AND length(FIRST_NAME) = 6;

#Q-19. Write an SQL query to print details of the Workers whose SALARY lies between 100000 and 500000.
SELECT * FROM WORKER WHERE SALARY between 100000 and 500000;

#Q-20. Write an SQL query to print details of the Workers who have joined in Feb’2014.
SELECT * FROM WORKER WHERE YEAR(JOINING_DATE) = 2014 AND MONTH(JOINING_DATE)= 02;

#Q-21. Write an SQL query to fetch the count of employees working in the department ‘Admin’.
SELECT COUNT(*) FROM WORKER WHERE DEPARTMENT LIKE 'ADMIN';

#Q-22. Write an SQL query to fetch worker names with salaries >= 50000 and <= 100000.
SELECT CONCAT(FIRST_NAME , " " , LAST_NAME) AS WORKER_NAME,SALARY FROM WORKER WHERE SALARY >=50000 AND SALARY <= 100000;

#Q-23. Write an SQL query to fetch the no. of workers for each department in the descending order.
SELECT DEPARTMENT,COUNT(*) AS NO_OF_WORKERS FROM WORKER GROUP BY DEPARTMENT ORDER BY NO_OF_WORKERS DESC;

#Q-24. Write an SQL query to print details of the Workers who are also Managers.
SELECT W.*
FROM WORKER W
INNER JOIN TITLE T
ON W.WORKER_ID = T.WORKER_REF_ID
WHERE T.WORKER_TITLE = 'MANAGER';

#Q-25. Write an SQL query to fetch duplicate records having matching data in some fields of a table.
SELECT WORKER_TITLE,AFFECTED_FROM,COUNT(*)
FROM TITLE
GROUP BY WORKER_TITLE,AFFECTED_FROM
HAVING COUNT(*) >1;

#Q-26. Write an SQL query to show only odd rows from a table.
SELECT * FROM Worker WHERE MOD (WORKER_ID, 2) <>0;

#Q-27. Write an SQL query to show only even rows from a table.
SELECT * FROM WORKER WHERE MOD(WORKER_ID,2) = 0;

#Q-28. Write an SQL query to clone a new table from another table.
CREATE TABLE WorkerClone LIKE Worker;

#Q-29. Write an SQL query to fetch intersecting records of two tables.
(SELECT WORKER_ID FROM WORKER)
INTERSECT
(SELECT WORKER_ID FROM WORKERCLONE);

#Q-30. Write an SQL query to show records from one table that another table does not have.
SELECT * FROM Worker
MINUS
SELECT * FROM Title;

#Q-31. Write an SQL query to show the current date and time.
SELECT NOW();

#Q-32. Write an SQL query to show the top n (say 10) records of a table.
SELECT * FROM WORKER
ORDER BY Salary DESC
LIMIT 10;

#Q-33. Write an SQL query to determine the nth (say n=5) highest salary from a table.
SELECT FIRST_NAME,SALARY FROM WORKER ORDER BY SALARY DESC LIMIT 4,1;



