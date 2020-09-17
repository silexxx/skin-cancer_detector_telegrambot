try:
    from PIL import Image
except ImportError:
    import Image



from fastai.vision import ImageDataBunch,get_transforms,imagenet_stats,cnn_learner,models,load_learner,open_image
from fastai.metrics import error_rate
import numpy as np


from telegram.ext.dispatcher import run_async
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
from telegram import Update, Bot, ParseMode
import os

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! \n\nWelcome to Skin Cancer Recognizer Bot. \n\nJust send a clear image to the bot and it will recognize the type of Skin cancer which are :\nActinic keratoses \nBasal cell carcinoma\nBenign keratosis\nDermatofibroma\nMelanocytic nevi\nMelanoma\nVascular lesions\n')

def search(bot, update):
    """Send reply of user's message."""
    photo_file = bot.get_file(update.message.photo[-1].file_id)
    photo_file.download('testing.jpeg')
    try:
        bs = 32
        path = "classes"

        np.random.seed(42)
        data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

        learn = cnn_learner(data, models.resnet34, metrics=error_rate).load("stage-1")
        learn.export()
        learn = load_learner("classes")


        cat, tensor, probs = learn.predict(open_image("testing.jpeg"))


        l=list(probs)
        a=tensor.__str__()
        a=int(a.strip("tensor""()"))
        l=list(probs)[a]
        l=l.__str__()
        b=float(l.strip("tensor""()"))
        if b>=0.9:
            update.message.reply_text('`'+str(cat)+'`',parse_mode=ParseMode.MARKDOWN,reply_to_message_id=update.message.message_id)
            print("prediction :")   
            print(cat)
        else:
            cat="sry I am not sure "
            update.message.reply_text('`'+str(cat)+'`',parse_mode=ParseMode.MARKDOWN,reply_to_message_id=update.message.message_id)
            print("prediction :")
            print("Not Sure")

    except Exception as e:
        update.message.reply_text(e)

def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)

def main():
    """Start the bot."""
    ocr_bot_token="1379744792:AAGwlnrcGy3a-E1m-9eYpdoPhVhrE_xAyKU"
    #ocr_bot_token=os.environ.get("BOT_TOKEN", "")
    updater = Updater(ocr_bot_token)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    # dp.add_handler(CommandHandler("donate", donate))
    # dp.add_handler(CommandHandler("contact", contact))
    dp.add_handler(MessageHandler(Filters.photo, search))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
