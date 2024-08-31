import { HStack, Box, Text } from '@chakra-ui/react'
import { Link } from 'react-router-dom'

interface Props {
  l: string,
  ll: string,
  r: string,
  rl: string
}
const FootLinks = ({ l, ll, r, rl }: Props) => {
  const hoverStyle = { color: 'tomato', textDecoration: 'underline' }

  return (
    <HStack justifyContent='space-between' pb={10}>
      <Box>
        <Link to={ll}>
          <Text color='gray'>Previous Chapter</Text>
          <Text as='b' color='gray.600' _hover={hoverStyle}>
            {`<- ${l}`}
          </Text>
        </Link>
      </Box>
      <Box>
        <Link to={rl}>
          <Text textAlign='right' color='gray'>Next Chapter</Text>
          <Text as='b' color='gray.600' _hover={hoverStyle}>
            {`${r} ->`}
          </Text>
        </Link>
      </Box>
    </HStack>
  )
}

export default FootLinks