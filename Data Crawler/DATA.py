from time import sleep

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_soup(url):
    sleep(1)
    page = requests.get(url, headers={'user-agent': 'Mozilla/5.0'})

    return BeautifulSoup(page.content, 'html.parser')


def get_games_links(soup):
    print("get_games_links")
    games_list = soup.find('div', class_='product_condensed')
    links = []
    for game in games_list.select('li[class*="product game_product"]'):
        links.append(game.a['href'])

    return links


def get_game_info(soup):
    print("get_game_info")
    title = soup.find('div', class_='product_title').find('h1').get_text()
    platform = soup.find('span', class_='platform').find('a').get_text().strip()
    summary = soup.find('span', class_='blurb blurb_expanded')
    if summary is not None:
        summary = summary.get_text().strip()
    else:
        summary = soup.find('span', itemprop='description')
        if summary is not None:
            summary = summary.get_text().strip()
        else:
            summary = np.nan
    release_date = soup.find('span', itemprop='datePublished').get_text().strip()
    developer = soup.find('li', class_='summary_detail developer')
    if developer is not None:
        developer = developer.find('span', class_='data').get_text().strip()
    else:
        developer = np.nan
    genre = []
    for g in soup.find('li', class_='summary_detail product_genre').find_all('span', class_='data'):
        genre.append(g.get_text().strip())

    rating = soup.find('li', class_='summary_detail product_rating')
    if rating is not None:
        rating = rating.find('span', class_='data').get_text().strip()
    else:
        rating = np.nan

    return title, platform, summary, release_date, developer, genre, rating


def get_reviews_overview(soup):
    print("get_reviews_overview")
    overview = soup.find('span', class_='desc').get_text().strip()
    reviews_count = soup.find('div', class_='score_distribution')
    if reviews_count is not None:
        reviews_count = reviews_count.find_all('span', class_='count')
        pos = reviews_count[0].get_text().strip()
        mixed = reviews_count[1].get_text().strip()
        neg = reviews_count[2].get_text().strip()
    else:
        pos = '0';
        mixed = '0';
        neg = '0'

    return overview, pos, mixed, neg


def get_reviews(soup, category='user'):
    print("get_reviews")
    names = [];
    dates = [];
    scores = [];
    texts = []

    reviews_list = soup.find('ol', class_=f'reviews {category}_reviews')
    if reviews_list is not None:
        for review in reviews_list.select(f'li[class*="review {category}_review"]'):
            if category == 'user':
                names.append(review.find('div', class_='name').get_text().strip())
            else:
                names.append(review.find('div', class_='source').get_text().strip())
            dates.append(review.find('div', class_='date').get_text().strip())
            scores.append(review.find('div', class_='review_grade').get_text().strip())
            exp = review.find('span', class_='blurb blurb_expanded')
            if exp is None:
                texts.append(review.find('div', class_='review_body').get_text().strip())
            else:
                texts.append(exp.get_text().strip())

    return names, dates, scores, texts


def create_reviews_df(critics_dict, dates_dict, scores_dict, texts_dict, titles, platforms):
    print("create_reviews_df")
    critics = [];
    dates = [];
    scores = [];
    texts = [];
    games = [];
    plats = []
    for k in critics_dict:
        critics += critics_dict[k]
        dates += dates_dict[k]
        scores += scores_dict[k]
        texts += texts_dict[k]
        games += [titles[k]] * len(critics_dict[k])
        plats += [platforms[k]] * len(critics_dict[k])

    return pd.DataFrame(
        {'critic': critics, 'date': dates, 'score': scores, 'text': texts, 'title': games, 'platform': plats},
        columns=['score', 'text', 'critic', 'date', 'title', 'platform'])


def main():
    print("entered main..")

    letters = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    # console = 'ps4'
    # console = 'xboxone'
    console = 'switch'

    games_links = []

    for letter in letters:
        soup = get_soup(f'http://www.metacritic.com/browse/games/title/{console}/{letter}')
        games_links += get_games_links(soup)
        p = soup.find('ul', class_='pages')
        if p is not None:
            pages_qty = len(p.find_all('li'))
            for page_num in range(1, pages_qty):
                soup = get_soup(f'http://www.metacritic.com/browse/games/title/{console}/{letter}?page={page_num}')
                games_links += get_games_links(soup)

    titles = {};
    platforms = {};
    summaries = {};
    release_dates = {}
    developers = {};
    genres = {};
    ratings = {};
    meta_scores = {}
    meta_overviews = {};
    meta_pos = {};
    meta_mixed = {}
    meta_neg = {};
    critics_names = {};
    critics_dates = {};
    critics_scores = {}
    critics_texts = {};
    user_scores = {};
    user_overviews = {}
    user_pos = {};
    user_mixed = {};
    user_neg = {};
    users_names = {}
    users_dates = {};
    users_scores = {};
    users_texts = {}

    print(type(games_links), len(games_links))

    for i in range(len(games_links)):
        link = games_links[i]
        # game summary section
        soup = get_soup(f'http://www.metacritic.com{link}')

        # game summary info
        title, platform, summary, release_date, developer, genre, rating = get_game_info(soup)
        titles[link] = title
        platforms[link] = platform
        summaries[link] = summary
        release_dates[link] = release_date
        developers[link] = developer
        genres[link] = genre
        ratings[link] = rating

        # critics reviews section
        soup = get_soup(f'http://www.metacritic.com{link}/critic-reviews')

        # critics reviews general info
        meta_score = soup.find('span', itemprop='ratingValue')
        if meta_score is not None:
            meta_scores[link] = meta_score.get_text().strip()
        else:
            meta_scores[link] = '0'
        overview, pos, mixed, neg = get_reviews_overview(soup)
        meta_overviews[link] = overview
        meta_pos[link] = pos
        meta_mixed[link] = mixed
        meta_neg[link] = neg
        # critics reviews
        names, dates, scores, texts = get_reviews(soup, 'critic')
        critics_names[link] = names
        critics_dates[link] = dates
        critics_scores[link] = scores
        critics_texts[link] = texts

        # users reviews section
        soup = get_soup(f'http://www.metacritic.com{link}/user-reviews')

        # users reviews general info
        user_scores[link] = soup.select('div[class*="metascore_w user large"]')[0].get_text().strip()
        overview, pos, mixed, neg = get_reviews_overview(soup)
        user_overviews[link] = overview
        user_pos[link] = pos
        user_mixed[link] = mixed
        user_neg[link] = neg
        # users reviews
        names, dates, scores, texts = get_reviews(soup)
        users_names[link] = names
        users_dates[link] = dates
        users_scores[link] = scores
        users_texts[link] = texts

        p = soup.find('ul', class_='pages')
        if p is not None:
            pages_qty = len(p.find_all('li'))
            for page_num in range(1, pages_qty):
                # sleep(randint(1, 3))
                soup = get_soup(f'http://www.metacritic.com{link}/user-reviews?page={page_num}')

                names, dates, scores, texts = get_reviews(soup)
                users_names[link] += names
                users_dates[link] += dates
                users_scores[link] += scores
                users_texts[link] += texts

    df = pd.DataFrame({'title': titles, 'platform': platforms, 'summary': summaries,
                       'release_date': release_dates, 'developer': developers, 'genre': genres,
                       'rating': ratings, 'meta_score': meta_scores, 'meta_overview': meta_overviews,
                       'meta_pos': meta_pos, 'meta_mixed': meta_mixed, 'meta_neg': meta_neg,
                       'user_score': user_scores, 'user_overview': user_overviews, 'user_pos': user_pos,
                       'user_mixed': user_mixed, 'user_neg': user_neg},
                      columns=['title', 'platform', 'developer', 'genre', 'rating', 'release_date',
                               'summary', 'meta_score', 'meta_overview', 'meta_pos', 'meta_mixed',
                               'meta_neg', 'user_score', 'user_overview', 'user_pos', 'user_mixed',
                               'user_neg']).reset_index(drop=True)

    df.to_csv(f'{console}_games.csv', index_label=False)

    df = create_reviews_df(critics_names, critics_dates, critics_scores, critics_texts, titles, platforms)
    df.to_csv(f'{console}_meta_reviews.csv', index=False)

    df = create_reviews_df(users_names, users_dates, users_scores, users_texts, titles, platforms)
    df.to_csv(f'{console}_user_reviews.csv', index_label=False)


if __name__ == '__main__':
    main()