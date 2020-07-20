## Love is Love Hackathon

### Problem statement

Love knows no gender and the LGBTQ (Lesbian, Gay, Bisexual, Transgender, and Queer) community is the epitome of this thought. During Pride Month, we(Hacker Earth) are here with another Machine Learning challenge to celebrate the impact and changes that they made globally.

You have been appointed as a social media moderator for your firm. Your key responsibility is to tag and categorize quotes that are uploaded during Pride Month on the basis of its sentiment, positive, negative, and random. Your task is to build a sophisticated Machine Learning model combining Optical Character Recognition (OCR) and Natural Language Processing (NLP) to assess sentiments of these quotes.

### Dataset

The dataset consists of quotes that are uploaded during Pride Month.

The benefits of practicing this problem by using unsupervised Machine Learning techniques are as follows:

This challenge encourages you to apply your unsupervised Machine Learning skills to build models that can assess sentiments of a quote.
This challenge helps you enhance your knowledge of OCR and NLP that are a part of the advanced fields of Machine Learning and artificial intelligence.
You are required to build a model that analyzes sentiments of a quote and classifies them into positive, negative, or random

### Detecting sentiments of a quote

You work as a social media moderator for your firm. Your key responsibility is to tag uploaded content (images) during Pride Month based on its sentiment (positive, negative, or random) and categorize them for internal reference and SEO optimization.

### Task
Your task is to build an engine that combines the concepts of OCR and NLP that accepts a .jpg file as input, extracts the text, if any, and classifies sentiment as positive or negative. If the text sentiment is neutral or an image file does not have any text, then it is classified as random.

### OCR implementation using Tesseract

#### What is OCR?

OCR is Optical Charcater Recgonition,Optical Character Recognition involves the detection of text content on images and translation of the images to encoded text that the computer can easily understand. An image containing text is scanned and analyzed in order to identify the characters in it. Upon identification, the character is converted to machine-encoded text.

__How is it really achieved?__ To us, text on an image is easily discernible and we are able to detect characters and read the text, but to a computer, it is all a series of dots.

The image is first scanned and the text and graphics elements are converted into a bitmap, which is essentially a matrix of black and white dots. The image is then pre-processed where the brightness and contrast are adjusted to enhance the accuracy of the process.

The image is now split into zones identifying the areas of interest such as where the images or text are and this helps kickoff the extraction process. The areas containing text can now be broken down further into lines and words and characters and now the software is able to match the characters through comparison and various detection algorithms. The final result is the text in the image that we're given.

__For this OCR project, I used the Python-Tesseract, or simply PyTesseract, library which is a wrapper for Google's Tesseract-OCR Engine.__

The Python-Tesseract is downloaded from the following location:
https://tesseract-ocr.github.io/tessdoc/

Pytesseract is a wrapper for Tesseract-OCR Engine. It is also useful as a stand-alone invocation script to tesseract, as it can read all image types supported by the Pillow and Leptonica imaging libraries, including jpeg, png, gif, bmp, tiff, and others. 

For Sentiment Analysis I have used VADER Sentiment Analysis.