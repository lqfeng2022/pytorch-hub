import { Heading, Center, Stack, Text, Image, AspectRatio } from '@chakra-ui/react'
import Cover from '../entities/chapterCover'

const Header = ({ cover }: { cover: Cover }) => {
  return (
    <>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {cover.name}
      </Heading>
      <Center py={3}>
        <Stack maxW='390px'>
          <Text as='i' textAlign='center' pb={3}>{`"${cover.quote}"`}</Text>
          <Text as='i' textAlign='center'>{cover.author}</Text>
        </Stack>
      </Center>
      <Stack py={5}>
        <AspectRatio ratio={ 4/3 }>
          <Image src={cover.image} />
        </AspectRatio>
        <Text as='b' fontSize='sm'>{cover.title}</Text>
        <Text as='i' fontSize='sm'>{cover.description}</Text>
      </Stack>
    </>
  )
}

export default Header