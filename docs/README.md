# Events

Migrated from Hugging Face Spaces.

## Original Space

- URL: https://huggingface.co/spaces/Chris4K/Events
- SDK: gradio

## Deployment

This space uses three-tier deployment:

1. **Private Monorepo** (development)
   - Location: `apps/huggingface/Events/`
   - All development happens here
   - Contains secrets and private configuration

2. **Public GitHub Repository**
   - URL: https://github.com/Ckal/Events
   - Synced via git subtree
   - Portfolio/showcase version
   - Deploy: `bash infra/deploy-public-repo.sh`

3. **Hugging Face Space**
   - URL: https://huggingface.co/spaces/Chris4K/Events
   - Auto-deploys from public GitHub via Actions
   - Live public demo

## Development

1. Make changes in `src/`
2. Test locally
3. Commit to private repo
4. Deploy to public: `bash infra/deploy-public-repo.sh`
5. GitHub Actions auto-pushes to HF Space

## Files

- `src/app.py` - Main application
- `src/requirements.txt` - Dependencies
- `build.sh` - Build script
- `app.yaml` - Application configuration
