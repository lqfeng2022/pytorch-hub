import { Stack, Text, Image } from '@chakra-ui/react'
import chatgpt from '../assets/chatgpt.webp'

const ChapterCover = () => {
  const cover = 
    { head: "Here's the ChatGPT introduction from OpenAI homepage.", 
      content: "When ChatGPT-3.5 came out on November 30, 2022, it was the first time that AI impressed us by its human-like text. It can help with answering questions, writing essays, creating content, tutoring in various subjects, and even having casual conversations. Each version has improved in terms of understanding context, handling more complex queries, and generating more accurate and relevant responses. In short, ChatGPT is like having a smart, versatile assistant right at your fingertips."
    }

  return (
    <Stack py={5}>
      <Image src={chatgpt} />
      <Text as='b'>{cover.head}</Text>
      <Text fontSize='sm' as='i'>{cover.content}</Text>
    </Stack>
  )
}

export default ChapterCover