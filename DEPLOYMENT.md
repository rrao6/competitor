# Competitor Monitor - Deployment Guide

## Quick Start: Vercel + Supabase

### Step 1: Create Supabase Database

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Go to **SQL Editor** and run the contents of `supabase_schema.sql`
3. Go to **Settings > Database** and copy the **Connection string (URI)**
   - It looks like: `postgresql://postgres:PASSWORD@HOST:5432/postgres`

### Step 2: Export Local Data to Supabase

```bash
# Set your Supabase connection string
export DATABASE_URL='postgresql://postgres:YOUR_PASSWORD@YOUR_HOST:5432/postgres'

# Run the export script
python export_to_supabase.py
```

### Step 3: Deploy to Vercel

1. Push your code to GitHub (the repo you already connected)
2. In Vercel dashboard, go to **Settings > Environment Variables**
3. Add:
   - `DATABASE_URL` = your Supabase connection string

4. Deploy:
```bash
vercel --prod
```

Or just push to main branch - Vercel will auto-deploy.

### Step 4: Verify Deployment

Visit your Vercel URL (e.g., `https://competitor.vercel.app`)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes (prod) | Supabase PostgreSQL connection string |
| `OPENAI_API_KEY` | For data collection | OpenAI API key |

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard (uses local SQLite)
python dashboard/app.py

# Run data collection
python run_radar.py --quick
```

---

## Project Structure for Vercel

```
/
├── api/
│   └── index.py          # Vercel entry point
├── dashboard/
│   ├── app.py            # Flask application
│   ├── static/           # CSS, images
│   └── templates/        # HTML templates
├── radar/                # Core logic
├── config/               # Configuration
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── supabase_schema.sql   # Database schema
```

---

## Troubleshooting

### "No module named 'radar'"
Make sure `api/index.py` adds the project root to `sys.path`.

### Database connection errors
- Check `DATABASE_URL` is set correctly in Vercel
- Make sure the connection string starts with `postgresql://` not `postgres://`

### Static files not loading
The Flask app serves static files. Vercel routes all requests through the Flask app.

---

## Updating Data

To run data collection and update Supabase:

```bash
# Set environment variables
export DATABASE_URL='your_supabase_url'
export OPENAI_API_KEY='your_openai_key'

# Run collection (data goes directly to Supabase)
python run_radar.py --quick
```

You can also set up a cron job or GitHub Action to run this periodically.
