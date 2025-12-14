
# ☁️ Deploying to AWS Elastic Beanstalk

This guide will walk you through deploying the **E-Commerce Smart Pricing Engine** to AWS using Elastic Beanstalk (EB). Elastic Beanstalk is the easiest way to deploy Python web applications on AWS as it handles the infrastructure (EC2, Load Balancers, Auto Scaling) for you.

## Prerequisites

1.  **AWS Account:** If you don't have one, sign up at [aws.amazon.com](https://aws.amazon.com/).
2.  **EB CLI (Elastic Beanstalk Command Line Interface):**
    *   **Windows:** `pip install awsebcli`
    *   **Mac/Linux:** `pip install awsebcli` --upgrade --user
    *   Verify installation: `eb --version`

## Steps to Deploy

### 1. Initialize Elastic Beanstalk
Open your terminal in the project directory (`D:\ML MAIN PROJECT`) and run:

```bash
eb init -p python-3.8 ecommerce-pricing-app
```
*   It may ask for your AWS Access Keys. You can get these from the *IAM* section of the AWS Console.
*   Select your preferred region (e.g., `us-east-1` or `ap-south-1`).

### 2. Create an Environment
This command creates the actual server environment (EC2 instances, etc.). This process takes about 5-10 minutes.

```bash
eb create ecommerce-pricing-env
```

### 3. Deploy the Application
Once the environment is created, deploy your code:

```bash
eb deploy
```

### 4. Access the Application
To open your deployed web app in the browser, simply run:

```bash
eb open
```

---

## 🔁 Updating the Application
If you make changes to your code (e.g., update `app.py` or retrain models), simply commit your changes in git and run:

```bash
eb deploy
```

## 🧹 Clean Up (Optional)
To avoid incurring charges when you are done testing, you can terminate the environment:

```bash
eb terminate ecommerce-pricing-env
```

## 🔧 Troubleshooting

*   **Instance Health:** Run `eb health` to check the status of your application.
*   **Logs:** Run `eb logs` to see error messages if the application fails to start.
*   **File Size:** AWS has limits on upload size. Ensure your `.gitignore` excludes large unnecessary files like `dataset/` if they aren't needed for inference (though your `requirements.txt` suggests standard usage, large CSVs should usually simply be S3-hosted if the app doesn't need them at runtime).
