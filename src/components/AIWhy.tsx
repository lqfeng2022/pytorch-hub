import { Box, Image, Stack, Text } from '@chakra-ui/react';
import aiwhy_img from '../assets/aiwhy_img.jpeg';

const AIWhy = () => {
  const topic = {
    name: "1.1 Why we should know about AI?",
    image: aiwhy_img,
    values: [
      { id: 0, 
        title: "1) ChatGPT.", 
        content: "When ChatGPT-3.5 came out on November 30, 2022, it was the first time that AI impressed us by its human-like text. If you're interested in ChatGPT and artificial intelligence, you should take this course. I am gonna lift the veil of mystery - ChatGPT, the most successful and powerful artificial intelligence product."
      },
      { id: 1, 
        title: "2) AI is everywhere!",
        content: "Is all around us. Even now, as you're watching this video. You can see it when you look at the camera or when you pick up your iPhone. You can feel it when you go to work,  when you go to restaurant, when you go to shop, when you chat with your friends. It's the world that has been pulled over your eyes, to blind you from the truth. The truth is that the AI is watching you! It knows where you go, what you eat, what you bought, and even what you talked with your friends.."
      },
      { id: 2,
        title: "3) BOOST your productivity at work..",
        content: "AI is awesome, especially when we wanna handle the repetitive tasks like data entry, managing schedules, summarizing information, and even analyzing papers. With AI, taking care of those boring tasks, we can focus more on higher-value tasks, like strategy analysis, employee engagement."
      },
      { id: 3,
        title: "4) AI powered Robot.",
        content: "What if you have an AI robot that can cook dinner, wash the dishes, clean the floor, take care of your parents, and even chat with you like an old friend. Would you want to pay for something like that?"
      }
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{topic.name}</Text>
      <Image py={5} src={topic.image}/>
      <Stack spacing={4}>
        {topic.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text>{p.content}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default AIWhy