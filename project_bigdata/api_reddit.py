import praw 
import csv
import time

# Initialisation de Reddit
reddit = praw.Reddit()

# Paramètres
subreddit_name = "Bitcoin"
num_posts = 1000  
num_comments_per_post = 5 
batch_size = 50  

sub = reddit.subreddit(subreddit_name)

rows = []
count = 0

for post in sub.new(limit=None):  
    post.comments.replace_more(limit=0) 
    top_comments = [c.body for c in post.comments[:num_comments_per_post]]
    
    rows.append({
        "id": post.id,
        "title": post.title,
        "author": str(post.author),
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "url": post.url,
        "top_comments": " ||| ".join(top_comments)
    })
    
    count += 1
    
    # Sauvegarde par lot de données
    if count % batch_size == 0:
        with open("reddit_posts.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if f.tell() == 0: 
                writer.writeheader()
            writer.writerows(rows)
        rows = [] 
        print(f"{count} posts récupérés...")
        time.sleep(1)  # pause pour respecter les limites de l'API de Reddit

    if count >= num_posts:
        break

# Sauvegarde finale
if rows:
    with open("reddit_posts.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)

print(f"{count} posts ont été récupérés")
