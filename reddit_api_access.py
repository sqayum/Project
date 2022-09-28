import pandas as pd
import praw
from prawcore.exceptions import PrawcoreException
import sqlite3
import os
from itertools import chain
from nlp_utils import print_completion_message


@print_completion_message(start_msg="\nInitializing database")
def _initialize_database(*, db_con):
    db_cur = db_con.cursor()

    db_cur.execute("""CREATE TABLE IF NOT EXISTS subreddits (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT NOT NULL,
                                issue TEXT NOT NULL,
                                subscribers INTEGER NOT NULL,
                                stance TEXT NOT NULL)""")
    db_con.commit()



    db_cur.execute("""CREATE TABLE IF NOT EXISTS posts (
                                id TEXT NOT NULL  PRIMARY KEY,
                                subreddit_id INTEGER NOT NULL,
                                author_id TEXT NOT NULL,
                                title TEXT,
                                score INTEGER NOT NULL,
                                upvote_ratio FLOAT(0,2) NOT NULL,
                                date INTEGER NOT NULL,
                                FOREIGN KEY (subreddit_id) REFERENCES subreddits(id)
                                FOREIGN KEY (author_id) REFERENCES users(id))""")
    db_con.commit()


    db_cur.execute("""CREATE TABLE IF NOT EXISTS comments(
                                id TEXT NOT NULL,
                                subreddit_id INTEGER NOT NULL,
                                post_id TEXT NOT NULL,
                                author_id TEXT NOT NULL,
                                body TEXT NOT NULL,
                                score INTEGER NOT NULL,
                                date INTEGER NOT NULL    ,
                                FOREIGN KEY (subreddit_id) REFERENCES subreddits(id),
                                FOREIGN KEY (post_id) REFERENCES posts(id),
                                FOREIGN KEY (author_id) REFERENCES users(id))""")
    db_con.commit()

    db_cur.execute("""CREATE TABLE IF NOT EXISTS users(
                                id TEXT NOT NULL,
                                subreddit_id INTEGER NOT NULL)""")
    db_con.commit()

    return db_con

@print_completion_message(start_msg="Extracting subreddits")
def _extract_subreddits(reddit_instance, *, subreddit_dict, db_con):
    db_cur = db_con.cursor()

    subreddits_extracted = db_cur.execute("""--sql
                                            SELECT COUNT(*)
                                            FROM subreddits
                                            """).fetchone()[0]
    if not subreddits_extracted:
        for stance in subreddit_dict:
            for issue in subreddit_dict[stance]:
                for subreddit_name in subreddit_dict[stance][issue]:
                    subreddit = reddit_instance.subreddit(subreddit_name.removeprefix("r/"))
                    subreddit_id = subreddit.id
                    num_subscribers = subreddit.subscribers

                    columns = (subreddit_name,
                                issue,
                                num_subscribers,
                                stance)

                    db_cur.execute("""--sql
                                        INSERT OR IGNORE INTO subreddits(name, issue, subscribers, stance)
                                        VALUES ( ?, ?, ?, ? )""",
                                        columns)
                    db_con.commit()
    return db_con


def _get_query(subreddit_name, current_issue):
    if subreddit_name == "r/AskTrumpSupporters":
        if current_issue == "climate":
            return "flair:Environment"
        elif current_issue == "immigration":
            return "flair:Immigration"
        elif current_issue == "healthcare":
            return "flair:Health Care"
    elif subreddit_name == "r/Conservative":
        return "flair:Flaired Users Only"
    elif subreddit_name == "r/ConservativesOnly":
        return "flair:Conservatives Only"
    elif subreddit_name == "r/Firearms":
        return "flair:Politics"
    elif subreddit_name == "r/prolife":
        return "NOT (flair:Moderator Message OR flair:Things Pro-Choicers Say)"
    elif subreddit_name == "r/prochoice":
        return "NOT (flair:Support OR flair:Things Anti-choicers Say OR flair:Resource/Abortion Funds Info OR flair:MOD ANNOUNCEMENT OR flair:Creators & Merch OR flair:Humor)"
    elif subreddit_name == "r/JoeBiden":
        if current_issue == "immigration":
            return "flair:Immigration"
    elif subreddit_name == "r/Political_Revolution":
        if current_issue == "immigration":
            return "flair:Immigration"
        elif current_issue == "healthcare":
            return "flair:Healthcare"
    else:
        return

def _get_posts(subreddit_instance, *, query=None, limit, sort, time_filter):
    if query is None:
        if sort == "hot":
            posts = subreddit_instance.hot(limit=limit)
        elif sort == "top":
            posts = subreddit_instance.top(limit=limit, time_filter=time_filter)
        elif sort == "new":
            posts = subreddit_instance.new(limit=limit)
        elif sort == "rising":
            posts = subreddit_instance.rising(limit=limit, time_filter=time_filter)
    else:
        posts = subreddit_instance.search(query=query, sort=sort, time_filter=time_filter)
    return posts

def _get_comments(post_instance, *, sort="top"):
    post_instance.comment_sort = sort
    post_instance.comments.replace_more()
    return post_instance.comments.list()


def _append_post_to_database(post, subreddit_id, *, db_con):
     db_cur = db_con.cursor()

     try:
        author_id = post.author.id
     except (PrawcoreException, AttributeError):
        try:
            author_id = post.author.name
        except (PrawcoreException, AttributeError):
            author_id = "<unkown>"

     post_attributes = (post.id,
                         subreddit_id,
                         author_id,
                         post.title,
                         post.score,
                         post.upvote_ratio,
                         post.created_utc)

     db_cur.execute("""--sql
                                INSERT OR IGNORE INTO posts( id, subreddit_id, author_id, title, score, upvote_ratio, date)
                                VALUES (?, ?, ?, ?, ?, ?, ?)""", post_attributes)
     db_con.commit()

     _append_user_to_database(author_id, subreddit_id, db_con=db_con)

     return db_con

def _append_comment_to_database(comment, post_id, subreddit_id, *, db_con):
    db_cur = db_con.cursor()

    try:
        author_id = comment.author.id
    except (PrawcoreException, AttributeError):
        try:
            author_id = comment.author.name
        except (PrawcoreException, AttributeError):
            author_id = "<unkown>"

    comment_attributes = (comment.id,
                              subreddit_id,
                              post_id,
                              author_id,
                              comment.body,
                              comment.score,
                              comment.created_utc)
    db_cur.execute("""--sql
                            INSERT OR IGNORE INTO comments(id, subreddit_id, post_id, author_id, body, score, date)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""", comment_attributes)
    db_con.commit()

    _append_user_to_database(author_id, subreddit_id, db_con=db_con)

    return db_con

def _append_user_to_database(author_id, subreddit_id, *, db_con):
    db_cur = db_con.cursor()

    author_attributes = (author_id,
                            subreddit_id)

    db_cur.execute("""--sql
                            INSERT OR IGNORE INTO users (id, subreddit_id)
                            VALUES (?, ?)""", author_attributes)
    db_con.commit()
    return db_con


def extract_data(reddit_instance, *, subreddit_dict,  database_path, max_post_requests, max_comment_requests, select_subreddits=None, sort="top", time_filter="all"):
    database_path = os.path.join(os.path.dirname(__file__), database_path)
    db_conn = sqlite3.connect(database_path)
    db_cursor = db_conn.cursor()

    db_conn = _initialize_database(db_con=db_conn)
    db_conn = _extract_subreddits(reddit_instance, subreddit_dict=subreddit_dict, db_con=db_conn)

    subreddits = db_cursor.execute("""SELECT id, name, issue FROM subreddits""").fetchall()
    if select_subreddits is not None:
        if all(isinstance(subreddit, int) for subreddit in select_subreddits):
            subreddits = filter(lambda subreddit: subreddit[0] in select_subreddits, subreddits)
        elif all(isinstance(subreddit, str) for subreddit in select_subreddits):
            subreddits = filter(lambda subreddit: subreddit[1] in select_subreddits, subreddits)
        else:
            raise TypeError("cannot mix types in <select_subreddits>")
    for subreddit in subreddits:
        (subreddit_id, subreddit_name, issue) = subreddit
        try:
            subreddit_instance = reddit_instance.subreddit(subreddit_name.removeprefix("r/"))
            query = _get_query(subreddit_name, issue)
            posts = _get_posts(subreddit_instance, query=query, limit=max_post_requests, sort=sort, time_filter=time_filter)
            i = 0
            for post in posts:
                db_conn = _append_post_to_database(post, subreddit_id, db_con=db_conn)
                print("\n------------------------------------------------------")
                print(f"[{subreddit_name} | {issue}]\t(id = {subreddit_id})\n------------------------------------------------------")
                print(f"Extracted Post [{i+1} / {max_post_requests}]")
                post_id = post.id
                comments = _get_comments(post)
                j = 0
                for comment in comments:
                    if j >= max_comment_requests:
                        break
                    db_conn = _append_comment_to_database(comment, post_id, subreddit_id, db_con=db_conn)
                    print(f"\tExtracted Comment [{j+1} / {max_comment_requests}]")
                    j += 1
                i += 1

        except PrawcoreException as exception:
            print(f"\nEXCEPTION: {exception}\n")
            continue



    return


def main():
    MAX_POST_REQUESTS = 50
    MAX_COMMENT_REQUESTS = 50
    DATABASE_PATH = "data/reddit_data.db"
    SUBREDDIT_DICT = {
                            "conservative":
                                {
                                    "gun_control": ["r/progun", "r/Firearms", "r/gunpolitics"],
                                    "abortion": ["r/prolife"],
                                    "climate": ["r/AskTrumpSupporters", "r/climateskeptics"],
                                    "immigration": ["r/AskTrumpSupporters", "r/ImmigrationReform"],
                                    "healthcare": ["r/AskTrumpSupporters"],
                                    "ideology": ["r/Conservative", "r/ConservativesOnly"],
                                    "party": ["r/Republican"]
                                },


                            "liberal":
                                {
                                    "gun_control": ["r/GunsAreCool", "r/guncontrol"],
                                    "abortion": ["r/prochoice"],
                                    "climate": ["r/climate", "r/ClimateOffensive"],
                                    "immigration": ["r/JoeBiden", "r/Political_Revolution"],
                                    "healthcare": ["r/MedicareForAll", "r/Political_Revolution"],
                                    "ideology": ["r/Liberal", "r/progressive"],
                                    "party": ["r/democrats"]
                                },
                        }



    reddit = praw.Reddit(site_name="sqayum")

    extract_data(reddit,
                    subreddit_dict=SUBREDDIT_DICT,
                    database_path=DATABASE_PATH,
                    max_post_requests=MAX_POST_REQUESTS,
                    max_comment_requests=MAX_COMMENT_REQUESTS,
                    select_subreddits=list(chain(range(2,11),range(12,25))),
                    sort="top",
                    time_filter="all")



if __name__ == "__main__":
        main()
