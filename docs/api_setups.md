# API Setup


## FRT Cloud Providers

Here are instructions to setup each provider. For demo purposes, it is easiest to start with Face++. The free tier of each provider is sufficient for small datasets.

### Amazon AWS
- Follow the steps [here](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) to create an account, then read the "Programmatic access" section of [this guide](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to get an access key ID and secret access key.

### Face++
- Create an account [here](https://console.faceplusplus.com/register) and follow [these steps](https://console.faceplusplus.com/documents/7079083) to obtain an API key and API secret.

### Microsoft Azure
- Reference the instructions [here](https://azure.microsoft.com/en-us/services/cognitive-services/face/#get-started) and [here](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/identity-client-library) to create an endpoint link and API key.

---

## Image scrapers

### Google News

If you choose to use Google News for image scraping, no additional steps are needed.

### Google Images

Follow the instructions [here](https://developers.google.com/custom-search/v1/overview) to create a Google Custom Search Engine (CSE) and obtain an API key. The CSE should be configured to search the entire web, not just a subset of sites.
Be aware that, Custom Search JSON API provides 100 search queries per day for free. If you need more, you may sign up for billing in the API Console. Additional requests cost $5 per 1000 queries, up to 10k queries per day.
