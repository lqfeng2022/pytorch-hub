import { Heading } from '@chakra-ui/react'

const ChapterTitle = () => {
  const title = {name: "Chapter 0: Artificial Intelligence"}

  return (
    <Heading size='xl' textAlign='center' py={3}>{title.name}</Heading>
  )
}

export default ChapterTitle