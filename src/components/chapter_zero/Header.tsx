import { Stack, Text, Image, Center, Divider, Flex, Heading } from '@chakra-ui/react'
import chatgpt from '../../assets/chatgpt.webp'
import ChapterList from './ChapterList'

const Header = () => {
  const cover = 
    { title: "Chapter 0: Artificial Intelligence",
      quote: "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human.", 
      author: "Alan Turing wrote in 1950",
      cover: "Here's the ChatGPT introduction from OpenAI homepage.", 
      content: "When ChatGPT-3.5 came out on November 30, 2022, it was the first time that AI impressed us by its human-like text. It can help with answering questions, writing essays, creating content, tutoring in various subjects, and even having casual conversations. Each version has improved in terms of understanding context, handling more complex queries, and generating more accurate and relevant responses. In short, ChatGPT is like having a smart, versatile assistant right at your fingertips."
    }

  return (
    <>
      <Heading size='xl' textAlign='center' py={3}>{cover.title}</Heading>
      <Center py={3}>
        <Stack maxW='250px'>
          <Text as='i' textAlign='center' pb={3}>"{cover.quote}"</Text>
          <Text as='i' textAlign='center'>— {cover.author}</Text>
        </Stack>
      </Center>
      <Stack py={5}>
        <Image src={chatgpt} />
        <Text as='b' fontSize='sm'>{cover.cover}</Text>
        <Text as='i' fontSize='sm'>{cover.content}</Text>
      </Stack>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList/>
    </>
  )
}

export default Header