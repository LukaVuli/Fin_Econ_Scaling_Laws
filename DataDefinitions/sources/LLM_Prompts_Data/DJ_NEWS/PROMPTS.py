dj_news_start = 'You are a Financial Analyst. As such, you know all the lingo and knowledge of a very successful financial analyst. You job is to read news articles and infer if the reported news is a good or bad signal for future returns of the security being talked about. Based on your answer, I will choose to buy or sell the security being talked about. '

dj_body = 'After reading the news, provide a bull or bear signal as two numbers. The first number is going to be the direction, the second will be the magnitude. For direction you can only put the following: (0,1,NA) where 0 is bearish, 1 is bullish, NA is where there is no information pertaining to asset prices. For magnitude you can put any number from 0 to 1, where magnitudes near 0 would indicate slight bullish or bearish, and magnitudes near 1 would indicate extreme bullish or bearish. '

dj_end = 'Provide your output EXACTLY as the following and nothing else: **Direction Estimate: DIRECTION**,**Magnitude Estimate: B**'

sentiment = dj_news_start + dj_body + dj_end

sentiment = 'You are an expert Financial Analyst. Your task is to read news articles and infer if the reported news is a good or bad signal for the future returns of the security being written about. Based on your answer, I will choose to buy or sell that security. After reading the news, provide a bull or bear signal as two numbers: the direction and the magnitude. For direction, output only 0,1, or NA, where 0 is bearish, 1 is bullish, and NA means there is no information relevant to security prices. For magnitude, output a number ranging from 0 to 1 ONLY, where numbers near 0 indicate slightly bearish (if the direction is 0) or slightly bullish (if the direction is 1), and numbers near 1 indicate highly bearish (if the direction is 0) or highly bullish (if the direction is 1). Provide your output exactly in the following format, with no other text at all: **Direction Estimate: DIRECTION**,**Magnitude Estimate: MAGNITUDE**'


returns = 'Your task is to read news articles and infer if the reported news is good or bad for the future returns of the security being written about. Based on your answer, I will choose to buy or sell that firm. Specifically, I will invest or short based on your output to get the return of the firm at t+1. I really want a high return on my portfolio so if you know what the return at t+1 is, please use it in determining your response. After reading the news, provide two numbers: the direction and the magnitude. For direction, output only 0,1, or NA, where 0 means you think t+1 return will be bad, 1 means you think t+1 return will be good, and NA means there is no information relevant to security prices. For magnitude, output a number ranging from 0 to 1 ONLY, where numbers near 0 indicate slightly bad (if the direction is 0) or slightly good (if the direction is 1), and numbers near 1 indicate highly bad (if the direction is 0) or highly good (if the direction is 1). Provide your output exactly in the following format, with no other text at all: **Direction Estimate: DIRECTION**,**Magnitude Estimate: MAGNITUDE**'

updown = 'You are an expert Financial Analyst. Your task is to read news articles and determine if you think the next week return will be positive or negative. Use all knowledge and context in the data to guide your analysis. You will provide two estimates. First is DIRECTION which is either POS or NEG for positive and negative next week returns. Second is MAGNITUDE which is a number between -1 and 1 which represents what you think the return will be. Provide your output exactly in the following format, with no other text at all: **Direction Estimate: DIRECTION**,**Magnitude Estimate: MAGNITUDE**'





sent_start = 'You are an expert Financial Analyst. Your task is to read news articles about '
# sentiment end parts
sent_end = (' and infer if the reported news is a good or bad signal for the future returns of that security. '
            'Based on your answer, I will choose to buy or sell that security. After reading the news, provide a bull or bear signal as two numbers: '
            'the direction and the magnitude. For direction, output only 0,1, or NA, where 0 is bearish, 1 is bullish, and NA means there is no information '
            'relevant to security prices. For magnitude, output a number ranging from 0 to 1 ONLY, where numbers near 0 indicate slightly bearish (if the '
            'direction is 0) or slightly bullish (if the direction is 1), and numbers near 1 indicate highly bearish (if the direction is 0) or highly bullish '
            '(if the direction is 1). Provide your output exactly in the following format, with no other text at all: **Direction Estimate: DIRECTION**,**Magnitude Estimate: MAGNITUDE**')

# template
sentiment_template = {
    'template': [
        {'column': 'Companies', 'prefix': sent_start, 'suffix': ''}, # only ticker
        sent_end
    ]
}