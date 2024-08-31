import { Heading, Center, Stack, Text, Image, AspectRatio } from '@chakra-ui/react'
import Cover from '../entities/Cover'

const Header = ({ cover }: { cover: Cover }) => {

  return (
    <>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {cover.title}
      </Heading>
      <Center py={3}>
        <Stack maxW='300px'>
          <Text as='i' textAlign='center' pb={3}>{`"${cover.quote}"`}</Text>
          <Text as='i' textAlign='center'>{`-- ${cover.author}`}</Text>
        </Stack>
      </Center>
      <Stack py={5}>
        <AspectRatio ratio={ 16/9 }>
          <Image src={cover.image} />
        </AspectRatio>
        <Text as='b' fontSize='sm'>{cover.name}</Text>
        <Text as='i' fontSize='sm'>{cover.descript}</Text>
      </Stack>
    </>
  )
}

export default Header