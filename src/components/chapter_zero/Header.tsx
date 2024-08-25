import { Center, Heading, Image, Stack, Text } from '@chakra-ui/react'
import covers from '../../data/covers'

const Header = () => {
  const cover = covers[0].cover
  const quoteText = `"${cover.quote}"`
  const authorText = `-- ${cover.author}`

  return (
    <>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {cover.title}
      </Heading>
      <Center py={3}>
        <Stack maxW='300px'>
          <Text as='i' textAlign='center' pb={3}>{quoteText}</Text>
          <Text as='i' textAlign='center'>{authorText}</Text>
        </Stack>
      </Center>
      <Stack py={5}>
        <Image src={cover.image} />
        <Text as='b' fontSize='sm'>{cover.name}</Text>
        <Text as='i' fontSize='sm'>{cover.descript}</Text>
      </Stack>
    </>
  )
}

export default Header