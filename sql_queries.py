import sqlite3


class DBManagement:
    def __init__(self, db_path="song_detection.db"):
        self.conn = sqlite3.connect(db_path, timeout=100)
        self.create_database()

    def search_fingerprint(self, fingerprint):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                SELECT * FROM fingerprints
                WHERE hash = ?
            """,
            (fingerprint,),
        )

        found = cursor.fetchall()

        return found

    def store_song(self, song_alias, song_name, youtube_id):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                INSERT INTO songs (id, title) VALUES (?, ?)
            """,
            (youtube_id, song_name),
        )

        cursor.execute(
            """
                INSERT INTO song_aliases (alias, song_id) VALUES (?, ?)
            """,
            (song_alias, youtube_id),
        )

        self.conn.commit()

    def store_fingerprint(self, youtube_id, fingerprint, timestamp):
        cursor = self.conn.cursor()

        cursor.executemany(
            """
                INSERT INTO fingerprints (song_id, hash, timestamp) VALUES (?, ?, ?)
            """,
            zip([youtube_id] * len(fingerprint), fingerprint, timestamp),
        )

        self.conn.commit()

    def check_if_id_exists(self, youtube_id):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                SELECT 1 FROM songs WHERE id = ?
            """,
            (youtube_id,),
        )

        return cursor.fetchone() is not None

    def get_song_name(self, youtube_id):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                SELECT title FROM songs
                WHERE id = ?
            """,
            (youtube_id,),
        )
        row = cursor.fetchone()
        title = None if row is None else row[0]

        return title

    def get_song_youtube_id(self, song_alias):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                SELECT song_id FROM song_aliases
                WHERE alias = ?
            """,
            (song_alias,),
        )
        row = cursor.fetchone()
        youtube_id = None if row is None else row[0]

        return youtube_id

    def create_database(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS songs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                );
            """
        )

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS song_aliases (
                    id INTEGER PRIMARY KEY,
                    alias TEXT NOT NULL,
                    song_id TEXT NOT NULL,
                    FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
                );
            """
        )

        cursor.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS song_alias ON song_aliases(alias);
            """
        )

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY,
                    song_id TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE,
                    UNIQUE (song_id, timestamp)
                );
            """
        )
        cursor.execute(
            """
                CREATE INDEX IF NOT EXISTS spectral_hashes ON fingerprints(hash);
            """
        )

        self.conn.commit()
